import whisper
import json

model = whisper.load_model("base")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='transcribe audio to text')
    parser.add_argument('--input_path', '-i', type=str, help='path to source file')
    parser.add_argument('--output_path', '-o', type=str, help='path for output file')
    parser.add_argument('--verbose', '-v', type=bool, default=False, help='True for verbose mode')
    args = parser.parse_args()

    result = model.transcribe(args.input_path, verbose=args.verbose)

    with open(args.output_path, 'w') as fp:
        json.dump(result['segments'], fp)