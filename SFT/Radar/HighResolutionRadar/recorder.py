import threading
import socket
import time
import numpy as np
import sys


class RecorderTCP(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, callback, ip, port=55158, *args, **kwargs):
        super(RecorderTCP, self).__init__(*args, **kwargs)
        self.daemon = True
        self._stop_event = threading.Event()
        self.address = (ip, port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(None)
        self.sock.connect(self.address)
        print('Recorder connection established with:', self.address)
        self.callback = callback
        self.frame = bytearray()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # read frame header and extract number of bytes for reading
            print('Waiting for next frame')
            header = self.sock.recv(10)
            print('Header received')
            self.frame = bytearray()  # <- new data frame
            nr_bytes = np.frombuffer(header[6:10], dtype=np.uint32)
            recv_bytes = 0
            t0 = time.time()
            while recv_bytes != nr_bytes:
                # data = self.sock.recv(4096)
                data = self.sock.recv(2097153) #bigger buffer for quicker receiving
                self.frame.extend(data)
                recv_bytes += len(data)
                sys.stdout.write(f'\rConn:{self.address} %d ' %recv_bytes)
            t1 = time.time()
            self.callback(self.frame)
            # print(f'\ntime: {t1 - t0} s - {round((recv_bytes * 8) / ((t1 - t0) * 1000000), 1)} M bit/s')


class Recorder(object):
    def __init__(self, master_ip, slave_ip=None, en_dual_eth=False):
        self.en_dual_eth = en_dual_eth

        # Master
        self.rec_master = RecorderTCP(self.master_callback, master_ip)
        self.rec_master.start()
        self.master_data = None

        # Salve
        if self.en_dual_eth:
            if slave_ip is None:
                raise TypeError('Salve IP is None, while dual Ethernet is used')
            self.rec_slave = RecorderTCP(self.slave_callback, slave_ip)
            self.rec_slave.start()
            self.slave_ready = True
            self.slave_data = None

    def master_callback(self, data):
        self.master_data = data

    def slave_callback(self, data):
        self.slave_data = data

    def reset(self):
        self.master_data = None
        self.slave_data = None

    def get_data(self):
        """ Wait for previous started threads to receive a frame """
        while self.master_data is None:
            pass
        # Extract data
        data = np.frombuffer(self.master_data, dtype=np.int16)

        if self.en_dual_eth:
            while self.slave_data is None:
                pass
            slave_data = np.frombuffer(self.slave_data, dtype=np.int16)
            data = np.append(data, slave_data, axis=0)

        return data







