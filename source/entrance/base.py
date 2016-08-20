import os
import codecs

def save_result(file_path, results):
    assert file_path is not None
    dir = ntpath.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    writer = codecs.open(file_path,"w+")
    for result in results:
        writer.write("%s\n" %"\t".join(map(str,result)))
    writer.close()

import ntpath

def get_in_out_files(input_path, output_path):
    input_files = []
    output_files = []
    if isinstance(input_path,list):
        input_files = input_path
        output_files = output_path
    elif isinstance(input_path, str):
        if os.path.isfile(input_path):
            input_files = [input_path]
            output_files = [output_path]
        elif os.path.isdir(input_path):
            input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]
            if input_path == output_path:
                output_files = [(f+".result") for f in input_files]
            else:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_files = [os.path.join(output_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]
        else:
            Exception("Test file can only be defined by a list of file paths or a directory paths!")
    else:
        raise Exception("Test file can only be defined by a list of file paths or a directory paths!")
    return input_files, output_files
