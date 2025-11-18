import librosa
import json
import sys
from audio import process_audio
from mei import mei_to_chroma, get_measure_timestamps, filter_measures_by_tstamp

def main():
    # Import MEI #
    meifile_path = './data/myTest.xml'
    try:
        with open(meifile_path, 'r', encoding='utf-8') as f:
            mei_xml = f.read()
    except FileNotFoundError:
        print(f'Error: MEI file not found: {meifile_path}')
        sys.exit(1)
    except Exception as e:
        print(f'Error reading MEI file: {e}')
        sys.exit(1)

    # Import audio #
    audiopath = './data/myTest.mp4'
    try:
        audio_data, frame_rate, trim_start, trim_end = process_audio(audiopath)
    except FileNotFoundError:
        print(f'Error: Audio file not found: {audiopath}')
        sys.exit(1)
    except Exception as e:
        print(f'Error processing audio file: {e}')
        sys.exit(1)
    
    ###########################################################################
    # Step 1: Calculate MEI chroma features
    ###########################################################################
    print("Step 1: Calculating MEI chroma features...")
    try:
        chroma_mei, id_to_chroma_index = mei_to_chroma(mei_xml)
        print(f"  ✓ MEI chroma shape: {chroma_mei.shape}")
    except Exception as e:
        print(f'Error calculating MEI chroma features: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

    ###########################################################################
    # Step 2: Calculate audio chroma features
    ###########################################################################
    print("Step 2: Calculating audio chroma features...")
    try:
        chroma_size = round(len(audio_data) / chroma_mei.shape[1])
        chroma_audio = librosa.feature.chroma_stft(y=audio_data, sr=frame_rate, hop_length=chroma_size)
        print(f"  ✓ Audio chroma shape: {chroma_audio.shape}")
    except Exception as e:
        print(f'Error calculating audio chroma features: {e}')
        sys.exit(1)

    ###########################################################################
    # Step 3: Calculate warping path (DTW)
    ###########################################################################
    print("Step 3: Calculating DTW warping path...")
    try:
        path = librosa.sequence.dtw(chroma_mei, chroma_audio)[1]
        path_dict = {key: value for (key, value) in path}
        print(f"  ✓ Warping path calculated with {len(path)} points")
    except Exception as e:
        print(f'Error calculating warping path: {e}')
        sys.exit(1)

    ###########################################################################
    # Step 4: Build dictionary {MEI id: time[seconds]}
    ###########################################################################
    print("Step 4: Building timestamp dictionary...")
    try:
        id_to_time = {}
        chroma_length = len(audio_data) / frame_rate / chroma_audio.shape[1]
        for id in id_to_chroma_index:
            id_to_time[id] = path_dict[id_to_chroma_index[id]] * chroma_length
            id_to_time[id] += trim_start / 1000  # Offset for trimmed audio in seconds
        print(f"  ✓ Mapped {len(id_to_time)} note IDs to timestamps")
    except Exception as e:
        print(f'Error building timestamp dictionary: {e}')
        sys.exit(1)

    ###########################################################################
    # Step 5: Save to JSON file
    ###########################################################################
    output_file = './data/tmp_output.json'
    print(f"Step 5: Saving results to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(id_to_time, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Results saved to {output_file}")
    except Exception as e:
        print(f'Error saving JSON file: {e}')
        sys.exit(1)

    ###########################################################################
    # Step 6: Further processing with tmp_output.json
    ###########################################################################
    print("Step 6: Processing results...")
    try:
        with open('./data/tmp_output.json', 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        mapping = get_measure_timestamps(loaded_data, mei_xml)
        filtered_mapping = filter_measures_by_tstamp(mapping)
        
    except Exception as e:
        print(f'Error in further processing: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()