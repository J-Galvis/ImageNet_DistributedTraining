import sys
import os
import pickle
import socket
import struct

def send_message(sock, message):
    """
    Envía un mensaje serializado con pickle a través del socket.
    
    Formato:
    [4 bytes: length (big-endian)] [message bytes]
    """
    data = pickle.dumps(message)
    length = len(data)
    
    header = struct.pack('!I', length)
    sock.sendall(header)
    sock.sendall(data)


def receive_message(sock):
    """
    Recibe un mensaje serializado con pickle a través del socket.
    
    Formato:
    [4 bytes: length (big-endian)] [message bytes]
    """
    header = sock.recv(4)
    if len(header) < 4:
        raise ConnectionError("Conexión cerrada por servidor")
    
    length = struct.unpack('!I', header)[0]
    
    data = b''
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            raise ConnectionError("Conexión cerrada durante recepción")
        data += chunk
    
    message = pickle.loads(data)
    return message
