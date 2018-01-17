import sys
import os
import datetime as dt

class log:

    __slots__ = ['_filepath', '_initime', '_printstr']

    def __init__(self, logobject, filepath='log.txt'):
        """
        initialize the log file; creation date, list of all important attributes
        """

        self._filepath = filepath
        self._initime = str(dt.datetime.now())  # creation time of file
        self._fileinit()
        self._printstr = ""

    def _fileinit(self):
        """
        actual file creation foo
        """
        pass

    def SimWatch(self, logobject, attrlist=None, watchlist=None):
        """
        log information about logobject. if list of attributes of logobject is
        given, log their development over time.
        """
        
