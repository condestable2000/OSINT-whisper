import whisper
import os

def transcribe(file_input):
    # load the model
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_input)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    #get name file from file_input
    route_file_input = file_input.split("/")
    name_file_input = route_file_input[-1]
    no_ext_file_input = name_file_input.split(".")
    name_file_output = no_ext_file_input[0] + ".txt"

    # save the transcription to a file
    with open(name_file_output, "w") as f:
        # detect the spoken language
        _, probs = model.detect_language(mel)
        f.write(f"Detected language: {max(probs, key=probs.get)}\n\n\n")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        f.write(result.text)   
    

def files_manager(rute):
    files = os.listdir(rute)
    for file in files:
        transcribe(file)

if __name__ == "__main__":
    # get inut parameters
    rute = input("Enter the rute of the audio files: ")
    files_manager(rute)
    print("The transcription is done")
