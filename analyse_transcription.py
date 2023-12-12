import os
import requests






#open the transction file whose name is transcription.txt and put first line in a string

def get_transcription():
    """
    Get the transcription from the transcription.txt file
    """
    with open("/transcription_output/transcription.txt", "r",encoding='utf-8') as f:
        lines = f.readlines()
        first_line = lines[0]

    return first_line

def analyse_turn_taking(text):
    """
    Analyse the turn taking in the conversation
    """
    #a faire
    return output



if __name__ == "__main__":
    text = get_transcription()
    print(text)
    output = analyse_turn_taking(text)
    print(output)




