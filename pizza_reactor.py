import typing as T
import select
from socket import socket, create_server


Data = bytes
Action = T.Union[
    T.Callable[[socket], None],
    T.Tuple[T.Callable[[socket, Data], None], str]
]

Mask = int

BUFFER_SIZE = 1024
ADDRESS = ("127.0.0.1", 12345)


class EventLoop:
    def __init__(self) -> None:
        self.writers = {}
        self.readers = {}

    def register_event(self, source: socket, event: Mask, action: Action) -> None:
        key = source.fileno()
        if event & select.POLLIN:
            self.readers[key] = (source, event, action)
        elif event & select.POLLOUT:
            self.writers[key] = (source, event, action)

    def unregister_event(self, source: socket) -> None:
        key = source.fileno()
        if self.readers.get(key):
            del self.readers[key]
        if self.writers.get(key):
            del self.writers[key]

    def run_forever(self) -> None:
        while True:
            readers, writers, _ = select.select(
                self.readers, self.writers, []
            )
            for reader in readers:
                source, event, action = self.readers.pop(reader)
                action(source)
            for writer in writers:
                source, event, _action = self.writers.pop(writer)
                action, msg = _action
                action(source, msg)


class Server:
    def __init__(self, event_loop: EventLoop) -> None:
        self.event_loop = event_loop
        try:
            print(f"Starting server on {ADDRESS}")
            self.server_socket = create_server(ADDRESS)
            self.server_socket.setblocking(False)
        except OSError:
            self.server_socket.close()
            print("Server stopped")

    def _on_accept(self, _: socket) -> None:
        try:
            conn, client_address = self.server_socket.accept()
        except BlockingIOError:
            return
        conn.setblocking(False)
        print(f"Connected to {client_address}")
        self.event_loop.register_event(conn, select.POLLIN, self._on_read)
        self.event_loop.register_event(self.server_socket, select.POLLIN, self._on_accept)

    def _on_read(self, conn: socket) -> None:
        try:
            data = conn.recv(BUFFER_SIZE)
        except BlockingIOError:
            return
        if not data:
            self.event_loop.unregister_event(conn)
            print(f"Connection with {conn.getpeername()} has been closed")
            conn.close()
            return
        message = data.decode().strip()
        print(f"message: {message}")
        self.event_loop.register_event(conn, select.POLLOUT, (self._on_write, message))

    def _on_write(self, conn: socket, message) -> None:
        try:
            order = int(message)
            responce = "Thx for ordering pizza"
        except ValueError:
            responce = "Wrong number of pizza, try again"
        print(f"Sending message to {conn.getpeername()}")
        try:
            conn.send(responce.encode())
        except BlockingIOError:
            return

        self.event_loop.register_event(conn, select.POLLIN, self._on_read)

    def start(self) -> None:
        print("Server listening for incoming connections")
        self.event_loop.register_event(self.server_socket, select.POLLIN, self._on_accept)


if __name__ == "__main__":
    event_loop = EventLoop()
    Server(event_loop).start()
    event_loop.run_forever()