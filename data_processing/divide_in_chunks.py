import os
import csv
import cv2

def get_nr_frames_video(video_path):
    cap = cv2.VideoCapture(video_path)
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return(nr_frames)


def divide_annotation_file(input_file, chunk_length, min_chunk_length, video_folder, output_file):

    input_file_name = os.path.basename(input_file)
    reader = csv.DictReader(open(input_file, 'r'), skipinitialspace=True)
    output_fields = reader.fieldnames[:]

    output_fields.append('chunk_nr_in_video')
    output_fields.append('sample_id_chunk') # video name + chunk
    output_fields.append('frame_start')
    output_fields.append('frame_end')
    output_fields.append('video_path_full')
    writer = csv.DictWriter(open(output_file, 'w'), fieldnames=output_fields, lineterminator=os.linesep)
    writer.writeheader()

    for row in reader:#Go through each file
        start = 0
        video_path_full = os.path.join(video_folder, row['Activity1'], row['VideoName'])
        end = get_nr_frames_video(video_path_full) - 1
        residual = (end - start) % chunk_length
        # this prevents chunks smaller than min_chunk_length if we split a annotation in multiple chunks
        # by skipping the start of the annotation
        # skipping the start is an arbitrary decision it asumes that the important part of the actions is more likely
        # at the end of an annotation than in the beginning
        if (end - start) / chunk_length > 1 and residual != 0 and residual < min_chunk_length:
            start += residual
            print('Moving the start of annotation "{}" by {} to prevent a short chunk. Length: {} Chunks: {}'.format(row['activity'], residual, (end - start), (end - start) / chunk_length))

        for chunk_id, chunk_start in enumerate(range(start, end, chunk_length)):
            chunk_end = min(chunk_start + chunk_length, end)
            if chunk_end - chunk_start < min_chunk_length:
                print('WARNING: Recorded a short chunk: ', chunk_end - chunk_start)
            output_row = row.copy()
            output_row['frame_start'] = chunk_start
            output_row['frame_end'] = chunk_end
            output_row['chunk_nr_in_video'] = chunk_id
            output_row['sample_id_chunk'] = row['VideoName']+"_" + str(chunk_id)
            output_row['video_path_full'] = video_path_full
            writer.writerow(output_row)

def main():

    split_file_path = "/cvhci/data/activity/Sims4ADL/SimsSplitsCompleteVideos.csv"
    output_file_path = "/cvhci/data/activity/Sims4ADL/SimsSplitsChunks.csv"
    video_folder = "/cvhci/data/dschneider/SimsVid/SimsVids/"
    
    divide_annotation_file(input_file = split_file_path, chunk_length = 90, min_chunk_length = 1, video_folder = video_folder, output_file = output_file_path)


if __name__ == "__main__":
    main()
