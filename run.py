from app import create_app

app = create_app()

if __name__ == '__main__':
    # For development, the reloader is useful.
    # For production, use Gunicorn as specified in the Dockerfile.
    app.run(debug=True, port=5000)