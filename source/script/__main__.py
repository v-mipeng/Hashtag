import os
import script

if __name__ == "__main__":
    # Build configuration object
    config = script.ScriptConfig()
    # Dump data from 10.141.209.3
    if False:
        downloader = script.MongoDumper(config)
        downloader.dump(2000000, save_to = config.train_path)
    # Tokenize post
    dir = os.path.dirname(config.train_path)
    file_name, _ = os.path.splitext(os.path.basename(config.train_path))

    new_file_path = os.path.join(dir, file_name+".tok")
    script.tokenize_text(config,config.train_path, new_file_path)
    # Change to UTH format
    read_from = new_file_path
    save_to = os.path.join(dir, file_name+".uth")
    script.format2UTHD(config, read_from = read_from, save_to = save_to)
