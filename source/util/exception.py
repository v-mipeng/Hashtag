class FileFormatError(Exception):
    def __init__(self, message = None):
        if message is None:
            message = "Wrong file format!"
        super(FileFormatError, self).__init__(message)


class FileNotExistError(Exception):
    def __init__(self, message = None):
        if message is None:
            message = "Cannot find file!"
        super(FileNotExistError, self).__init__(message)