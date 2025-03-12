import os
import json
import dotenv
import hashlib
import requests

from tqdm import tqdm
from time import sleep
from pathlib import Path
from natsort import natsorted
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_zenodo_token() -> str:
    """Load and return the Zenodo token from environment variables."""
    dotenv.load_dotenv()
    return os.environ['ZENODO_TOKEN']


@contextmanager
def create_retry_session(allowed_methods=None, total=15, backoff_factor=1.1, status_forcelist=None):
    """
    Create a requests.Session with retry logic.
    Defaults to allowed_methods=["PUT"] and typical transient error status codes.
    """
    allowed_methods = ["PUT"] if allowed_methods is None else allowed_methods
    status_forcelist = [500, 502, 503, 504] if status_forcelist is None else status_forcelist

    session = requests.Session()

    retries = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        yield session
    finally:
        session.close()


class ZenodoClient:
    BASE_URL = "https://zenodo.org/api/deposit/depositions"

    def __init__(self, token: str, deposition_id: int, session: requests.Session):
        self.token = token
        self.deposition_id = deposition_id
        self.session = session
        self.deposition = self.get_deposition()
        self.bucket_url = self.deposition['links']['bucket']

    def get_deposition(self) -> dict:
        """Retrieve and return deposition details from Zenodo."""
        url = f"{self.BASE_URL}/{self.deposition_id}"
        params = {"access_token": self.token}
        response = self.session.get(
            url,
            params=params,
            json={},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def reorder_files(self, filelist: list) -> dict:
        """
        Reorder the files in the deposition by specifying the desired order.

        Parameters:
          file_ids: A list of file IDs in the desired order. For example:
                    [ {'id': "21fedcba-98..."}, {'id': "12345678-9a..."}, ]

        Returns:
          A dict representing the updated file ordering response from Zenodo.

        Reference:
          https://github.com/zenodo/developers.zenodo.org/blob/master/source/includes/resources/deposit-files/_sort.md
          https://github.com/zenodo/zenodo/blob/482ee72ad501cbbd7f8ce8df9b393c130d1970f7/tests/unit/deposit/test_api_simpleflow.py#L176
          https://github.com/zenodo/zenodo/blob/482ee72ad501cbbd7f8ce8df9b393c130d1970f7/tests/unit/deposit/test_api_files.py#L160
        """
        url = self.deposition['links']['files']
        headers = {
            'Authorization': f"Bearer {self.token}",
            'Content-Type': "application/json",
            'Accept': "application/json",
        }
        response = self.session.post(url, json=filelist, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_file_list(self) -> list:
        """Retrieve and return the list of files in the Zenodo bucket."""
        response = self.session.get(
            self.deposition['links']['files'],
            headers={"Authorization": f"Bearer {self.token}"},
        )
        response.raise_for_status()
        data = response.json()
        return data

    def upload_file(self, file_path: Path, timeout: int = 30) -> dict:
        """Upload the file to Zenodo using a progress bar and return the JSON response."""
        file_size = file_path.stat().st_size
        upload_url = f"{self.bucket_url}/{file_path.name}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
        }
        with file_path.open("rb") as fd:
            with tqdm.wrapattr(fd, "read", total=file_size, desc=file_path.name, unit="B",
                               unit_scale=True) as wrapped_fd:
                response = self.session.put(
                    url=upload_url,
                    data=wrapped_fd,
                    timeout=timeout,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute the MD5 checksum of the file at file_path.
    Returns the hex digest string.
    """
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main(zenodo_id: int, file_to_upload: Path):
    zenodo_token = load_zenodo_token()

    local_md5 = compute_md5(file_to_upload)

    with create_retry_session(allowed_methods=["PUT"]) as session:
        client = ZenodoClient(zenodo_token, zenodo_id, session)
        for file in client.get_file_list():
            if file["filename"] == file_to_upload.name:
                print(f"Checksums: Local: {local_md5}, Zenodo: {file['checksum']}")
                if local_md5 == file["checksum"]:
                    raise FileExistsError("File already exists in Zenodo bucket")
                else:
                    print("File already exists in Zenodo bucket but MD5 checksum mismatch")
        result = client.upload_file(file_to_upload)
        print("Upload successful:", result)

    result_md5 = result.get("checksum", "").replace("md5:", "")

    if local_md5 == result_md5:
        print("MD5 checksum matches:", local_md5)
    else:
        raise ValueError(f"MD5 checksum mismatch! Local: {local_md5}, Zenodo: {result_md5}")


def sort_files_on_zenodo(zenodo_id: int):
    zenodo_token = load_zenodo_token()

    with create_retry_session() as session:
        client = ZenodoClient(zenodo_token, zenodo_id, session)
        files = client.get_file_list()
        print(files)
        files = natsorted(files, key=(lambda x: x['filename']))

    with create_retry_session() as session:
        client = ZenodoClient(zenodo_token, zenodo_id, session)
        result = client.reorder_files([{'id': file['id'], 'filename': file['filename']} for file in files])
        return result


if __name__ == "__main__":
    zenodo_id = 15006666

    # sort_files_on_zenodo(zenodo_id); exit(0);

    files_to_upload = natsorted(Path(".").parent.glob("results/001_big/run/**/*_*.pt"))

    for file_to_upload in files_to_upload:
        print(f"Uploading {file_to_upload} to Zenodo...")
        sleep(2)
        try:
            main(zenodo_id, file_to_upload)
        except FileExistsError:
            print("File already exists in Zenodo bucket. Skipping upload.")
