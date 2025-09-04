Semantic Question Similarity API
A flexible, Flask-based API designed to perform semantic analysis on a list of questions. It leverages multiple AI providers (including Google Gemini, OpenAI, and DeepSeek) to create text embeddings and determine similarity. The API provides endpoints to check if a new question is a duplicate and to group existing questions into semantically similar clusters.

The application is containerized with Docker and docker-compose for easy setup and deployment.

Features
Multi-Provider AI: Dynamically choose between different AI providers (e.g., Gemini, OpenAI, DeepSeek) for embedding and reasoning tasks.

JWT Authentication: Secure endpoints using JSON Web Tokens.

Semantic Similarity Check: Compares a new question against a list from a URL to find matches.

Question Grouping: Clusters a list of questions into groups based on semantic similarity.

Rate Limiting: Protects against brute-force attacks on the login endpoint.

Highly Configurable: Key settings like API keys, model names, and application behavior are managed via environment variables.

Dockerized: Comes with Dockerfile and docker-compose.yml for a consistent and isolated production-ready environment.

🚀 Setup and Installation
Follow these steps to get the application running on your local machine using Docker.

Prerequisites
Git

Docker Desktop (with Docker Compose)

Step 1: Clone the Repository
First, clone the project repository to your local machine.

git clone <your-repository-url>
cd <repository-directory>

Step 2: Configure Environment Variables
The application is configured using a .env file. Create this file in the root of the project directory by copying the example values below.

You must replace the placeholder values for the API keys and password with your own credentials. You only need to provide keys for the services you intend to use.

# --- Core Credentials (Required) ---
ADMIN_USERNAME="webapp_admin"
ADMIN_PASSWORD="replace_with_a_strong_and_secure_password"
JWT_SECRET_KEY="a_long_random_secure_string_for_jwt_signing"

# --- AI Provider API Keys (Fill in the ones you will use) ---
GOOGLE_API_KEY="REPLACE_WITH_YOUR_GOOGLE_API_KEY"
OPENAI_API_KEY="REPLACE_WITH_YOUR_OPENAI_API_KEY"
DEEPSEEK_API_KEY="REPLACE_WITH_YOUR_DEEPSEEK_API_KEY"

# --- AI Model Name Configuration ---
# Set the specific model names for each provider and task
GEMINI_EMBEDDING_MODEL="text-embedding-004"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

GEMINI_REASONING_MODEL="gemini-1.5-flash"
OPENAI_REASONING_MODEL="gpt-4o"
DEEPSEEK_REASONING_MODEL="deepseek-chat"

# --- Application Behavior Settings ---
JWT_EXPIRATION_HOURS=8
SIMILARITY_THRESHOLD=0.85

# --- CORS Settings (Update for your frontend) ---
CORS_ORIGINS="http://localhost:3000,[https://your-production-frontend.com](https://your-production-frontend.com)"

Step 3: Build and Run with Docker Compose
Build the Docker images and run the services using docker-compose. This will also start the memcached service required for rate limiting.

# Build and run the containers in detached mode
docker-compose up --build -d

The API will now be running and accessible at http://localhost:5000.

API Request/Response Formats
POST /login
Request

{
  "username": "webapp_admin",
  "password": "your_strong_password"
}

Response

{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

POST /check_similarity
Request

{
  "questions_url": "[https://example.com/api/questions.json](https://example.com/api/questions.json)",
  "question": "How do I set up a Docker container?",
  "embedding_provider": "gemini",
  "reasoning_provider": "openai"
}

Response (Match Found)

{
  "response": "yes",
  "matched_questions": [
    {
      "Question": "What are the steps to configure a Docker container?",
      "Answer": "First, you need to create a Dockerfile..."
    }
  ]
}

Response (No Match)

{
  "response": "no"
}

POST /group_similar_questions
Request

{
  "questions_url": "[https://example.com/api/questions.json](https://example.com/api/questions.json)",
  "embedding_provider": "openai"
}

Response (Groups Found)

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

Response (No Groups)

{
  "response": "no"
}