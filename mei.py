"""
MEI processing utilities for audio-to-score alignment.

Large portions of this code are adapted from the score-tube project:
https://github.com/cemfi/score-tube/blob/master/backend/mei.py
Original authors: cemfi
Licensed under: MIT License

Modifications:
- Adapted for use with meico library integration
- Added measure timestamp extraction functionality
- Modified chromagram generation for DTW alignment
"""

import math
import jpype
from jpype.types import *
import librosa
import numpy as np
from lxml import etree


def _start_jvm_if_needed():
    """Starte JVM falls noch nicht gestartet."""
    if not jpype.isJVMStarted():
        jpype.startJVM(
            "-Djava.awt.headless=true",
            "--add-opens=java.xml/com.sun.org.apache.xerces.internal.parsers=ALL-UNNAMED",
            "--add-opens=java.xml/com.sun.org.apache.xerces.internal.jaxp=ALL-UNNAMED",
            "--add-opens=java.xml/com.sun.org.apache.xerces.internal.util=ALL-UNNAMED",
            classpath=['./meico.jar'],
            convertStrings=False
        )


def mei_to_chroma(mei_xml):
    """Generate timestamps for all notes and rests of the MEI file using the Java framework meico.

    :param mei_xml: String containing the MEI XML data.
    :return: MEI string with included timestamps.
    """
    _start_jvm_if_needed()
    
    mei = jpype.JPackage('meico').mei.Mei(mei_xml, False)  # read in MEI data in meico
    mei.addIds()  # add unique IDs in case they are omitted
    mei.exportMsm(720, True, False)  # generate timestamps with ppq=720, no channel 10, no cleanup
    meico_xml = mei.toXML()
    
    # Konvertiere Java String zu Python String
    if isinstance(meico_xml, jpype.JString):
        meico_xml = str(meico_xml)
    
    # Stelle sicher, dass es als bytes vorliegt für lxml
    if isinstance(meico_xml, str):
        meico_xml = meico_xml.encode('utf-8')

    return _meico_to_chroma(meico_xml)


def _meico_to_chroma(meico_xml):
    """Convert meico MEI XML data to chromagram."

    :param meico_xml: String containing the MEI XML data, including additional data calculated by meico.
    :return: Chromagram and dictionary with MEI IDs to chromagram indices.
    """
    # parse XML data
    parser = etree.XMLParser(collect_ids=False)
    meico_xml = etree.fromstring(meico_xml, parser=parser)

    shortest_duration = np.inf
    highest_date = 0
    notes_and_rests = {}
    id_to_chroma_index = {}

    # iterate through all rests and notes and extract pitch, date, and duration
    for elem in meico_xml.xpath('//*[local-name()="note"][@midi.dur]|//*[local-name()="rest"][@midi.dur]'):
        identifier = elem.get('{http://www.w3.org/XML/1998/namespace}id')

        pitch = (int(float(elem.get('pnum'))) if elem.tag == '{http://www.music-encoding.org/ns/mei}note' else None)
        date = float(elem.get('midi.date'))
        dur = float(elem.get('midi.dur'))

        # put all relevant data in dictionary for easier lookup (xpath calls are expensive!)
        notes_and_rests[identifier] = {}
        notes_and_rests[identifier]['pitch'] = pitch
        notes_and_rests[identifier]['date'] = date
        notes_and_rests[identifier]['dur'] = dur

        shortest_duration = min(shortest_duration, dur)  # save shortest note duration for grid resolution of chroma matrix
        highest_date = max(highest_date, date + dur)  # save overall length of score file

    # init chromagram matrix with zeros
    chroma_matrix = np.zeros((12, int(highest_date / shortest_duration)), dtype=np.float16)

    # add chroma feature for every note to matrix
    for elem in notes_and_rests:
        note_or_rest = notes_and_rests[elem]
        begin = math.floor(note_or_rest['date'] / shortest_duration)
        id_to_chroma_index[elem] = begin
        if note_or_rest['pitch'] is not None:  # only notes (pitch != None)
            end = math.ceil((note_or_rest['date'] + note_or_rest['dur']) / shortest_duration)
            end = min(end, chroma_matrix.shape[1])  # limit end to maximum dimension in case of rounding errors
            chroma_matrix[note_or_rest['pitch'] % 12, begin:end] += 1

    chroma_matrix = librosa.util.normalize(chroma_matrix)  # normalize each chroma feature independently

    return chroma_matrix, id_to_chroma_index

def get_measure_timestamps(output_json, mei_xml):
    """Map MEI note IDs to their corresponding measure numbers based on timestamps.

    :param output_json: Dictionary mapping MEI note IDs to timestamps in seconds.
    :param mei_xml: String containing the MEI XML data.
    :return: Dictionary mapping MEI note IDs to measure numbers.
    """
    # parse XML data
    parser = etree.XMLParser(collect_ids=False)
    mei_tree = etree.fromstring(mei_xml.encode('utf-8'), parser=parser)
    result = []

    # Definiere Namespaces
    namespaces = {
        'xml': 'http://www.w3.org/XML/1998/namespace',
        'mei': 'http://www.music-encoding.org/ns/mei'
    }

    # Erstelle eine Kopie der Keys, um während der Iteration zu iterieren
    for key in list(output_json.keys()):
        measure_elems = mei_tree.xpath(
            f'//*[@xml:id="{key}"]/ancestor::mei:measure',
            namespaces=namespaces
        )
        
        # Prüfe, ob ein measure-Element gefunden wurde
        if not measure_elems:
            print(f'Warning: No measure found for element with ID "{key}" - skipping')
            continue
        
        measure_elem = measure_elems[0]
        measure_number = measure_elem.get('n')
        measure_id = measure_elem.get('{http://www.w3.org/XML/1998/namespace}id')
        
        # Prüfe, ob measure_number und measure_id existieren
        if measure_number is None or measure_id is None:
            print(f'Warning: Measure for element "{key}" has no number or ID - skipping')
            continue
            
        result.append({
            'measure_number': measure_number, 
            'measure_id': measure_id, 
            'timestamp_sec': output_json[key]
        })
    
    return result

def filter_measures_by_tstamp(measure_timestamps):
    """Filter MEI measures based on timestamps.
    Behält für jede measure_id nur den Eintrag mit dem frühesten timestamp_sec.

    :param measure_timestamps: Liste von Dictionaries mit measure_number, measure_id und timestamp_sec
    :return: Gefilterte Liste mit eindeutigen Measures (frühester Timestamp pro measure_id).
    """
    seen = {}
    
    for line in measure_timestamps:
        measure_id = line['measure_id']
        timestamp = line['timestamp_sec']
        
        # Speichere nur den Eintrag mit dem kleinsten Timestamp für diese measure_id
        if measure_id not in seen or timestamp < seen[measure_id]['timestamp_sec']:
            seen[measure_id] = line
    
    # Sortiere das Ergebnis nach Taktnummer und dann nach Timestamp
    result = sorted(seen.values(), key=lambda x: (int(x['measure_number']), x['timestamp_sec']))

    print(result)
    
    return result