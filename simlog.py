import sys
import os
import datetime as dt


class Log:
    """I am a docstring."""

    __slots__ = ['_path', '_initime', '_printstr', '_watchData', '_listenData']

    def __init__(self, log_path='eval/'):
        """Initialize the log file; creation date, list of all important attributes."""
        self._path = log_path
        self._initime = str(dt.datetime.now())  # creation time of file
        self.__foldersetup()
        self._printstr = ""
        self._watchData = []
        self._listenData = []

    def __fileinit(self):
        """Actual file creation foo."""
        pass

    def __folder_setup(self):
        """Folder setup for logfile(s)."""
        try:
            os.mkdir(self._path)

        except FileExistsError:
            try:
                os.mknod('report.txt')
                with open('report.txt', 'w') as f:
                    f.write("logfile date: "+self._initime)

            except FileExistsError:
                print(":: logfile already exists - to overwrite, call TBD")

    def watch(self, watch_object, attrlist=None, watchlist=None):
        """
        Log information about logobject.

        if list of attributes of logobject is given, log their
        development over time, i.e. everytime Watch is called, append certain information to the
        _watchData attribute.
        """
        attr = [watch_object.at for at in attrlist]
        self._watchData.append([attr, watchlist])

    def listen(self, system_parameters, finalize=False):
        """
        Listen to certain system parameters.

        like cpu usage, used RAM, runtime of script, etc. and
        append to _listenData attribute.
        If finalize is set, append data a last time, then calculate averages and maybe other stuff.
        """
        self._listenData.append(system_parameters)

        if(finalize):
            # calculate some averages -> data
            # self._toFile(data)
            pass

    def __to_File(self, data, separator='\t'):
        """'private' method to write some data so a log file with given separator."""
