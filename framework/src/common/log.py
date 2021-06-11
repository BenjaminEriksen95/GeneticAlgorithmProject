"""
Log object
"""

# Own modules
from . import ILog


__author__ = "Benjamin Eriksen"


class Log(ILog):
    def __init__(self):
        self.listeners = dict()

    def add_listener(self, id):
        file = open("logs/" + id + ".csv", "at")
        self.listeners[id] = file

    def add_entry(self, id, entry):
        self.listeners[id].write(entry)

    def remove_listener(self, id):
        self.listeners[id] = None
