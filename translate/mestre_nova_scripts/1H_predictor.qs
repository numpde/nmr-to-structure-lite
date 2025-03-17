 /*globals TextStream, File, MoleculePlugin, Molecule, Application, MessageBox, NMRPredictor, molecule, gc, Custom1DCsvConverter, performance*/

// ADAPTED FROM:
// https://github.com/rxn4chemistry/nmr-to-structure/blob/806de515f36dd61a874f9a713c4a3a3ce50d68a1/src/nmr_to_structure/nmr_generation/mestre_nova_scripts/1H_predictor.qs

function predict_1H(smiles, out_path) {
    // Create new document and predict spectrum
    dw = Application.mainWindow.newDocument();
    mol = get_mol_from_smiles(dw, smiles);

    // Predict NMR
    specId1H = NMRPredictor.predict(mol, "1H");
    var spec = nmr.activeSpectrum();

    // Process NMR
    process_1H_NMR(spec);

    // Multiplets
    var multiplets = spec.multiplets();

    // Save data
    var format = settings.value("Custom1DCsvConverter/CurrentFormat", "{ppm}{tab}{real}{tab}{imag}");
    // Replace {tab} with comma for CSV format
    save_spectrum_multiplet(smiles, dw.curPage()['items'], out_path, format.replace(/\{tab\}/g, ","), 6, false, multiplets);

    dw.destroy();
}

function process_1H_NMR(nmr_spectrum) {
    mainWindow.doAction("nmrAutoPeakPicking");
    peaks = nmr_spectrum.peaks();

    for (i = 0; i < peaks.count; i++) {
        peak = peaks.at(i);
        if (peak.type == Peak.Types.Solvent) {
            peak.type = Peak.Types.Compound;
            peak.flags = '2151682176';
        }
    }

    mainWindow.doAction("nmrMultipletsAuto");
}

function get_mol_from_smiles(aDocWin, aSMILES) {
    var mol_id = molecule.importSMILES(aSMILES);
    var mol = new Molecule(aDocWin.getItem(mol_id));
    return mol;
}

function save_spectrum_multiplet(smiles, aPageItems, aFilename, aFormat, aDecimals, aReverse, multiplets) {
    "use strict";
    var file, strm, i, line;

    try {
        file = new File(aFilename);
        if (file.open(File.WriteOnly)) {
            strm = new TextStream(file);

            // Write SMILES and a separator line
            strm.writeln('# SMILES: ' + smiles);

            // Metadata
            strm.writeln('# Multiplets: ' + multiplets.count);
            strm.writeln('# Date: ' + (new Date()).toISOString());
            strm.writeln('# MestReNova version: ' + Application.version);

            // Write multiplets header
            strm.writeln("category,centroid,delta,j_values,nH,rangeMax,rangeMin");

            // Write multiplets data
            for (i = 0; i < multiplets.count; i++) {
                multiplet = multiplets.at(i);

                try{
                    // Extract J-coupling values
                    var jList = multiplet.jList(); // Returns an array of J-coupling objects
                    var jValues = [];
                    for (var j = 0; j < jList.count; j++) {
                        jValues.push(jList.at(j));
                    }
                    var jValuesStr = jValues.join('_'); // Concatenate J-values with '_'
                } catch(e){
                    strm.writeln("# Exception found: {0}".format(e));
                    var jValuesStr = "None";
                }

                if (jValuesStr == "") {
                    jValuesStr = "None";
                }

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
        print("Exception found: {0}".format(e));
    } finally {
        file.close();
    }

    Application.quit();
}
