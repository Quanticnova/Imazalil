import sys
import os
import datetime as dt

class log:

    __slots__ = ['_filepath', '_initime', '_printstr', '_watchData', '_listenData']

    def __init__(self, logobject, filepath='log.txt'):
        """
        initialize the log file; creation date, list of all important attributes
        """

        self._filepath = filepath
        self._initime = str(dt.datetime.now())  # creation time of file
        self._fileinit()
        self._printstr = ""
        self._watchData = []
        self._listenData = []

    def _fileinit(self):
        """
        actual file creation foo
        """
        pass

    def Watch(self, watch_object, attrlist=None, watchlist=None):
        """
        log information about logobject. if list of attributes of logobject is given, log their
        development over time, i.e. everytime Watch is called, append certain information to the
        _watchData attribute.
        """
        attr = [watch_object.at for at in attrlist]
        self._watchData.append([attr, watchlist])

    def Listen(self, system_parameters, finalize=False):
        """
        listen to certain system parameters like cpu usage, used RAM, runtime of script, etc. and
        append to _listenData attribute.
        If finalize is set, append data a last time, then calculate averages and maybe other stuff.
        """
        self._listenData.append(system_parameters)

        if(finalize):
            # calculate some averages -> data
            # self._toFile(data)

    def _toFile(self, data, separator='\t'):
        """
        'private' method to write some data so a log file with given separator.
        """
