

-----

# Semantic Question Similarity API

A Flask-based API designed to perform semantic analysis on a list of questions. It leverages Google's Generative AI (`text-embedding-004`) to create text embeddings and determine similarity. The API provides endpoints to check if a new question is a duplicate and to group existing questions into semantically similar clusters.

The application is containerized with Docker for easy setup and deployment.

-----

## Features

  * **JWT Authentication**: Secure endpoints using JSON Web Tokens.
  * **Semantic Similarity Check**: Compares a new question against a list from a URL to find matches.
  * **Question Grouping**: Clusters a list of questions into groups based on semantic similarity.
  * **Rate Limiting**: Protects against brute-force attacks on the login endpoint.
  * **Configurable**: Key settings like JWT expiration, CORS origins, and similarity thresholds can be configured via environment variables.
  * **Dockerized**: Comes with a `Dockerfile` for a consistent and isolated production-ready environment.

-----

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine using Docker.

### Prerequisites

  * Git
  * Docker Desktop

### Step 1: Clone the Repository

First, clone the project repository to your local machine.

```sh
git clone <your-repository-url>
cd <repository-directory>
```

### Step 2: Configure Environment Variables

The application is configured using a `.env` file. Create this file in the root of the project directory by copying the example values below.

**You must replace the placeholder values for `GOOGLE_API_KEY` and `ADMIN_PASSWORD` with your own credentials.**

```.env
# --- Replace with your actual production credentials ---
GOOGLE_API_KEY="REPLACE_WITH_YOUR_REAL_GOOGLE_API_KEY"
ADMIN_USERNAME="webapp_admin"
ADMIN_PASSWORD="replace_with_a_strong_and_secure_password"
JWT_SECRET_KEY="1d1e4f5a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1"

# --- Application Behavior Settings ---
JWT_EXPIRATION_HOURS=8
SIMILARITY_THRESHOLD=0.90
EMBEDDING_MODEL_NAME="text-embedding-004"

# --- CORS Settings ---
# A comma-separated list of your frontend application URLs.
CORS_ORIGINS="https://your-production-frontend.com,https://your-staging-frontend.com"
```

### Step 3: Build and Run the Docker Container

Build the Docker image and then run the container. The `--env-file` flag will load the variables you just configured.

```sh
# 1. Build the Docker image
docker build -t similarity-api .

# 2. Run the Docker container
docker run -d -p 5000:5000 --env-file .env --name similarity-api-container similarity-api
```

The API will now be running and accessible at `http://localhost:5000`.

-----

## API Endpoint Details

All requests and responses are in JSON format. The protected endpoints require a JWT token to be sent in the `Authorization` header as a Bearer token.

### Authentication

#### `POST /login`

Authenticates a user and provides a JWT access token needed for protected endpoints.

  * **Request Body**:
    ```json
    {
      "username": "webapp_admin",
      "password": "your_strong_password"
    }
    ```
  * **Success Response (200 OK)**:
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
    ```
  * **Error Response (401 Unauthorized)**:
    ```json
    {
      "error": "Bad username or password"
    }
    ```

### Core API

**Note**: For the following endpoints, you must include the access token in the request header:
`Authorization: Bearer <your_access_token>`

#### `POST /check_similarity`

Checks if a new question is semantically similar to any question from a list hosted at a given URL.

  * **Request Body**:
    ```json
    {
      "questions_url": "https://example.com/api/questions.json",
      "question": "How do I set up a Docker container?"
    }
    ```
  * **Success Response (200 OK)**:
      * If a similar question is found:
        ```json
        {
          "response": "yes",
          "matched_questions": [
            {
              "Question": "What are the steps to configure a Docker container?",
              "Answer": "First, you need to create a Dockerfile..."
            }
          ]
        }
        ```
      * If no similar question is found:
        ```json
        {
          "response": "no"
        }
        ```

#### `POST /group_similar_questions`

Fetches all questions from a URL and groups them into clusters of semantically similar questions.

  * **Request Body**:
    ```json
    {
      "questions_url": "https://example.com/api/questions.json"
    }
    ```
  * **Success Response (200 OK)**:
      * If groups (of 2 or more) are found:
        ```json
        {
          "response": "yes",
          "matched_groups": [
            [
              { "Question": "Question A1", "Answer": "..." },
              { "Question": "Question A2", "Answer": "..." }
            ],
            [
              { "Question": "Question B1", "Answer": "..." },
              { "Question": "Question B2", "Answer": "..." },
              { "Question": "Question B3", "Answer": "..." }
            ]
          ]
        }
        ```
      * If no groups are formed:
        ```json
        {
          "response": "no"
        }
        ```
