/*globals TextStream, File, MoleculePlugin, Molecule, Application, MessageBox, NMRPredictor, molecule, gc, Custom1DCsvConverter, performance*/

// ADAPTED FROM:
// https://github.com/rxn4chemistry/nmr-to-structure/blob/806de515f36dd61a874f9a713c4a3a3ce50d68a1/src/nmr_to_structure/nmr_generation/mestre_nova_scripts/1H_predictor.qs

function predict_1H(smiles, out_path) {
    // Create a new empty document in Mnova
    dw = Application.mainWindow.newDocument();

    // Generate a molecule object from the SMILES string
    mol = get_mol_from_smiles(dw, smiles);

    // Predict the 1H NMR spectrum for the molecule
    specId1H = NMRPredictor.predict(mol, "1H");

    // Get the active spectrum after prediction
    var spec = nmr.activeSpectrum();

    // Process the predicted spectrum: peak picking and multiplet analysis
    process_1H_NMR(spec);

    // Retrieve the multiplet data from the processed spectrum
    var multiplets = spec.multiplets();

    // Retrieve and prepare format settings for exporting data
    var format = settings.value("Custom1DCsvConverter/CurrentFormat", "{ppm}{tab}{real}{tab}{imag}");

    // Save the spectrum multiplets to a file, replacing tabs with commas for CSV output
    save_spectrum_multiplet(smiles, dw.curPage()['items'], out_path, format.replace(/\{tab\}/g, ","), 6, false, multiplets);

    // Clean up: close the document to free resources
    dw.destroy();
}

function process_1H_NMR(nmr_spectrum) {
    // Automatically pick peaks in the NMR spectrum
    mainWindow.doAction("nmrAutoPeakPicking");

    // Iterate through peaks to modify solvent peaks into regular compound peaks
    peaks = nmr_spectrum.peaks();
    for (i = 0; i < peaks.count; i++) {
        peak = peaks.at(i);
        if (peak.type == Peak.Types.Solvent) {
            peak.type = Peak.Types.Compound;
            // Specific flags setting for the peak type; origin unknown
            // 2151682176 = 0b10000000010000000001000010000000
            peak.flags = '2151682176';
        }
    }

    // Automatically identify and annotate multiplets
    mainWindow.doAction("nmrMultipletsAuto");
}

function get_mol_from_smiles(aDocWin, aSMILES) {
    // Import the molecule from SMILES into the document window
    var mol_id = molecule.importSMILES(aSMILES);

    // Create a Molecule object from the imported structure
    var mol = new Molecule(aDocWin.getItem(mol_id));

    return mol;
}

function save_spectrum_multiplet(smiles, aPageItems, aFilename, aFormat, aDecimals, aReverse, multiplets) {
    "use strict";
    var file, strm, i, line;

    try {
        // Open a new file for writing
        file = new File(aFilename);
        if (file.open(File.WriteOnly)) {
            strm = new TextStream(file);

            // Write initial metadata lines to the output file
            strm.writeln('# SMILES: ' + smiles);
            strm.writeln('# Multiplets: ' + multiplets.count);
            strm.writeln('# Date: ' + (new Date()).toISOString());
            strm.writeln('# MestReNova version: ' + Application.version);

            // Write the CSV header for multiplet data
            strm.writeln("category,centroid,delta,j_values,nH,rangeMax,rangeMin");

            // Loop through each multiplet and write its data
            for (i = 0; i < multiplets.count; i++) {
                multiplet = multiplets.at(i);

                try {
                    // Extract all J-coupling values associated with the multiplet
                    var jList = multiplet.jList();
                    var jValues = [];
                    for (var j = 0; j < jList.count; j++) {
                        jValues.push(jList.at(j));
                    }
                    // Concatenate J-coupling values into a string, separated by underscores
                    var jValuesStr = jValues.join('_');
                } catch (e) {
                    // In case of an error extracting J-couplings, log it and default to "None"
                    strm.writeln("# Exception found: {0}".format(e));
                    var jValuesStr = "None";
                }

                // Default J-values string if empty
                if (jValuesStr == "") {
                    jValuesStr = "None";
                }

                // Format and write one line of multiplet information
                var line = multiplet.category + ',' +
                    multiplet.centroid + ',' +
                    multiplet.delta + ',' +
                    jValuesStr + ',' +
                    multiplet.nH + ',' +
                    multiplet.rangeMax + ',' +
                    multiplet.rangeMin;

                strm.writeln(line);
            }
        }
    } catch (e) {
        // Handle unexpected exceptions during file operations
        print("Exception found: {0}".format(e));
    } finally {
        // Always close the file stream properly
        file.close();
    }

    // Close Mnova when done (important for batch processing)
    Application.quit();
}
