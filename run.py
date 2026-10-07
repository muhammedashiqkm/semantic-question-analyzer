from app import create_app
from config import Config

app = create_app()

if __name__ == '__main__':
    # Development only. In production Gunicorn serves this, and binds the same
    # PORT - see the Dockerfile.
    #
    # HOST is 127.0.0.1 here on purpose: a development run should not be
    # reachable from the rest of the network. The container sets 0.0.0.0,
    # because a container that listens only on its own loopback cannot be
    # reached even by the host that published its port.
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
