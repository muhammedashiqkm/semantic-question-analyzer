
# Semantic Question Similarity API

## API Endpoint Details

**Note**: Protected endpoints require an `Authorization: Bearer <your_access_token>` header.

### `POST /login`

Authenticates a user and returns a JWT access token.

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

-----

### `GET /health`

Provides a simple health check endpoint to confirm the API is running.

  * **Success Response (200 OK)**:
    ```json
    {
      "status": "api_healthy"
    }
    ```

-----

### `POST /check_similarity`

Checks if a new question is semantically similar to any question from a list hosted at a given URL. This endpoint first validates the new question's quality before performing the comparison.

  * **Request Body**:

    ```json
    {
      "questions_url": "https://example.com/api/questions.json",
      "question": "How do I set up a Docker container?",
      "embedding_provider": "openai",
      "reasoning_provider": "gemini"
    }
    ```

      * **`embedding_provider`** (optional): Specifies the AI service for generating embeddings (e.g., `gemini`, `openai`). Defaults to the server's configuration if omitted.
      * **`reasoning_provider`** (optional): Specifies the AI service for initial question validation (e.g., `gemini`, `openai`, `deepseek`). Defaults to the server's configuration if omitted.

  * **Success Response (200 OK): Match Found**

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

  * **Success Response (200 OK): No Match Found**

    ```json
    {
      "response": "no"
    }
    ```

  * **Success Response (200 OK): No Existing Questions**
    (Occurs when the `questions_url` is valid but returns an empty list).

    ```json
    {
      "response": "no",
      "reason": "No existing questions to compare against."
    }
    ```

  * **Error Responses**:

      * **400 Bad Request (Validation Error)**: The request body is missing required fields or types are incorrect.
        ```json
        {
          "question": ["Missing data for required field."]
        }
        ```
      * **400 Bad Request (Poor Quality Question)**: The AI reasoning model determined the input question was invalid or low-quality.
        ```json
        {
          "error": "Invalid or poor-quality question provided."
        }
        ```
      * **404 Not Found**: The provided `questions_url` could not be reached or the resource was invalid.
        ```json
        {
          "error": "Resource not found at URL or could not be parsed."
        }
        ```
      * **500 Internal Server Error (Config Error)**: The specified providers do not have a corresponding model configured on the server.
        ```json
        {
          "error": "Server configuration error: model name not found for a specified provider."
        }
        ```
      * **503 Service Unavailable**: The external AI service (for embeddings or reasoning) failed or timed out.
        ```json
        {
          "error": "AI service is unavailable or returned an error."
        }
        ```

-----

### `POST /group_similar_questions`

Fetches all questions from a URL, generates embeddings, and groups them into clusters of semantically similar questions using hierarchical clustering.

  * **Request Body**:

    ```json
    {
      "questions_url": "https://example.com/api/questions.json",
      "embedding_provider": "gemini",
      "reasoning_provider": "gemini"
    }
    ```

      * **`embedding_provider`** (optional): Specifies the AI service for generating embeddings (e.g., `gemini`, `openai`). Defaults to the server's configuration if omitted.

  * **Success Response (200 OK): Groups Found**
    (Only groups containing more than one question are returned).

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
          { "Question": "Question B2", "Answer": "..." }
        ]
      ]
    }
    ```

  * **Success Response (200 OK): No Groups Found**
    (This occurs if questions were found, but none were similar enough to form a group).

    ```json
    {
      "response": "no"
    }
    ```

  * **Success Response (200 OK): Not Enough Questions**
    (This occurs if the URL returned 0 or 1 question, as clustering requires at least two).

    ```json
    {
      "response": "no",
      "reason": "Not enough questions to form a group."
    }
    ```

  * **Error Responses**:

      * **400 Bad Request (Validation Error)**: The request body is missing `questions_url`.
        ```json
        {
          "questions_url": ["Missing data for required field."]
        }
        ```
      * **500 Internal Server Error (Config Error)**: The specified embedding provider does not have a model configured.
        ```json
        {
          "error": "Server configuration error: model name not found for the specified provider."
        }
        ```
      * **500 Internal Server Error (Embedding Failure)**: Embeddings could not be generated.
        ```json
        {
          "error": "Failed to generate embeddings."
        }
        ```
      * **503 Service Unavailable**: The external AI embedding service failed or timed out.
        ```json
        {
          "error": "AI service is unavailable or returned an error."
        }
        ```