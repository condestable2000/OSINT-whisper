import whisper
import os

def transcribe(file_input, model):

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

    print('route: ',route_file_input[:-1])
    print('name_file_input: ',name_file_input)
    print('no_ext_file_input: ',no_ext_file_input)
    print('name_file_output: ',name_file_output)
    
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
    print(result.text  + "\n\n\n")

def files_manager(route, model):
    files = os.listdir(route)
    for file in files:
        print("file: ", route + "/" + file )
        transcribe(route + "/" + file, model)

if __name__ == "__main__":
    # get inut parameters
    route = input("Enter the rute of the audio files: ")
    # load the model
    model = whisper.load_model("medium") # "large" | "medium" | "small" | "base" | "tiny"
    files_manager(route, model)
    print("The transcription is done!!!")
