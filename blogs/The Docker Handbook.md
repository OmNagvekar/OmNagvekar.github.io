# The Docker Handbook

These notes are based on a [YouTube video](https://www.youtube.com/watch?v=kTp5xUtcalw) and cover core Docker concepts. You will learn how to build and manage images and containers, write detailed Dockerfiles and Docker Compose files, and get tips on choosing the right base images.
Download the this [Github Repo](https://github.com/K8sAcademy/Fundamentals-HandsOn) repository as it contains course material and sample projects for running docker containers.

---

## 1. Docker Overview

**Docker** is a platform that helps you build, ship, and run applications in lightweight, portable containers. Containers package an application together with its dependencies to ensure it runs consistently on any system.

_Key Benefits:_

- **Portability:** Run the same container on your local machine, in the cloud, or on any server.
- **Isolation:** Each container runs in its own environment, ensuring applications don’t interfere with each other.
- **Efficiency:** Containers use fewer resources than full virtual machines.

---

## 2. Working with Docker Images

### 2.1 Building Images

- **Basic Build:**
    
    ```bash
    docker build -t hello-docker .
    ```
    
    _Tags the image as “hello-docker” using the Dockerfile in the current directory._
    
- **Custom Dockerfile Location:**
    
    ```bash
    docker build -t my-app -f /path/to/Dockerfile .
    ```
    
- **Tagging an Existing Image:**
    
    ```bash
    docker tag existing-image:latest myusername/my-app:1.0
    ```
    

### 2.2 Pulling Images

- **Docker Login:**
	- 
	```bash
	docker login -u <username> -p <password>
	``` 

	- _This command Defaults to Docker hub for login_
	
- **Pull an Image from a Repository:**
    
    ```bash
    docker pull nginx
    ```
    
    _This command downloads the official Nginx image from Docker Hub._
- **Pushing an Image to the Repository:**

	```bash
	docker push username/image:tag
	```

---

## 3. Managing Docker Containers

### 3.1 Listing Containers

- **List Running Containers:**
    
    ```bash
    docker ps
    ```
    
- **List All Containers (Including Stopped):**
    
    ```bash
    docker ps -a
    ```
    

### 3.2 Running Containers

- **Interactive Mode (Shell):**
    
    ```bash
    docker run -it ubuntu
    ```
    
- **Detached Mode (Background):**
    
    ```bash
    docker run -d nginx
    ```
    
- **With Resource Limits:**
    
    ```bash
    docker run --memory="256m" nginx    # Limit memory usage  
    docker run --cpus=".5" nginx          # Limit CPU usage
    ```
    

### 3.3 Managing Running Containers

- **Attach a Shell to a Container:**
    
    ```bash
    docker run -it nginx /bin/bash
    docker container exec -it <container_name> bash
    ```
    
- **Run Temporary Containers with `--rm`:**
    
    ```bash
    docker run --rm -d -p 3000:3000 --name webserver my_test_express_app:latest
    ```
    
- **Start, Stop, and Kill Containers:**
    
    ```bash
    docker start <container_name>
    docker stop <container_name>
    docker kill <container_name>
    ```
    

### 3.4 Removing Images and Containers

- **Remove a Local Image:**
    
    ```bash
    docker rmi <IMAGE_ID>
    ```
    
- **Remove a Container:**
    
    ```bash
    docker rm <container_name>
    ```
    
- **Remove All Stopped Containers:**
    
    ```bash
    docker rm $(docker ps -a -q)
    ```
    
- **Force Remove All Images:**
    
    ```bash
    docker rmi --all --force
    ```
    
- **Clean Up Unused Resources:**
    
    ```bash
    docker system prune
    docker system prune -a  # Also removes unused images
    ```
    

### 3.5 Inspecting and Debugging

- **Inspect an Image:**
    
    ```bash
    docker image inspect <image_name>
    ```
    
- **Run with Port Mapping and Custom Name:**
    
    ```bash
    docker run --publish 80:80 --name webserver nginx
    ```
    
    > **Note:**
    > 
    > - `--publish` maps a host port to the container’s port.
    > - `--name` assigns a custom name to the container.
    

---

## 4. Writing Dockerfiles

A **Dockerfile** is a text file with instructions to build a Docker image. It must be named exactly `Dockerfile` (with no extension). Below is a detailed explanation of many Dockerfile commands.

### 4.1 Essential Dockerfile Commands

#### 4.1.1 Base Image

- **FROM:**  
    Sets the base image. Choose an image based on your needs:
    
    - **Alpine:** Lightweight, minimal footprint (good for production if you don’t need many tools).
    - **Ubuntu/Debian:** More comprehensive, includes more utilities (ideal for development or if you need extra packages).
    
    ```dockerfile
    FROM node:alpine
    # or
    FROM node:14  # Node on a Debian/Ubuntu base
    ```
    

#### 4.1.2 Adding Files

- **COPY:**  
    Copies files from the host to the container.
    
    ```dockerfile
    COPY . /app
    ```
    
- **ADD:**  
    Similar to COPY but can also extract compressed files and fetch files from URLs.
    
    ```dockerfile
    ADD archive.tar.gz /app
    ```
    
    _Use COPY when you simply need to copy files; use ADD for extra functionality like auto-extraction._

#### 4.1.3 Setting Environment

- **ENV:**  
    Sets environment variables inside the container.
    
    ```dockerfile
    ENV NODE_ENV production
    ```
    
- **ARG:**  
    Defines variables that users can pass at build-time.
    
    ```dockerfile
    ARG PORT=3000
    ```
    

#### 4.1.4 Setting the Working Directory

- **WORKDIR:**  
    Specifies the working directory for subsequent commands.
    
    ```dockerfile
    WORKDIR /app
    ```
    

#### 4.1.5 Installing Dependencies and Running Commands

- **RUN:**  
    Executes commands during the build process.
    
    ```dockerfile
    RUN npm install
    ```
    

#### 4.1.6 Exposing Ports and Declaring Volumes

- **EXPOSE:**  
    Informs Docker that the container listens on a specified network port.
    
    ```dockerfile
    EXPOSE 3000
    ```
    
- **VOLUME:**  
    Creates a mount point for persistent or shared data.
    
    ```dockerfile
    VOLUME ["/data"]
    ```
    

#### 4.1.7 Defining Container Behavior

- **CMD:**  
    Specifies the default command to run when the container starts.
    
    ```dockerfile
    CMD ["node", "app.js"]
    ```
    
- **ENTRYPOINT:**  
    Sets a command that is run every time the container starts, making it harder to override.
    
    ```dockerfile
    ENTRYPOINT ["node", "./app.js"]
    ```
    

#### 4.1.8 Additional Instructions

- **LABEL:**  
    Adds metadata to the image.
    
    ```dockerfile
    LABEL maintainer="youremail@example.com"
    ```
    
- **HEALTHCHECK:**  
    Configures a command to check the health of the container.
    
    ```dockerfile
    HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost/ || exit 1
    ```
    
- **USER:**  
    Sets the user to run the container as.
    
    ```dockerfile
    USER node
    ```
    
- **STOPSIGNAL:**  
    Sets the signal that will be sent to the container to exit.
    
    ```dockerfile
    STOPSIGNAL SIGTERM
    ```
    
- **SHELL:**  
    Overrides the default shell used for RUN commands.
    
    ```dockerfile
    SHELL ["/bin/bash", "-c"]
    ```
    

### 4.2 Detailed Dockerfile Examples

#### Example 1: Node.js Application

```dockerfile
# Use Node.js on Alpine Linux as the base image for a lightweight setup
FROM node:alpine

# Add metadata to the image
LABEL maintainer="yourname@example.com"

# Set build-time variables
ARG PORT=3000

# Set environment variables
ENV NODE_ENV production
ENV PORT $PORT

# Set the working directory inside the container
WORKDIR /app

# Copy package files and install dependencies
COPY package.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE $PORT

# Define a health check for the container
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:$PORT/ || exit 1

# Specify the default command to run the app
CMD ["node", "app.js"]
```

#### Example 2: Python Application

```dockerfile
# Use an official Python runtime as the base image (using slim for a smaller footprint)
FROM python:3.9-slim

# Add metadata
LABEL maintainer="yourname@example.com"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

### 4.3 How to Choose Base Images

- **Alpine:**
    - **Pros:** Very small, fast to download, and secure.
    - **Cons:** May lack some libraries or utilities needed for development.
- **Ubuntu/Debian:**
    - **Pros:** Rich in utilities and packages, good for development.
    - **Cons:** Larger image size.
- **Language-specific images:**  
    Often, official images like `node`, `python`, or `golang` come in different variants (e.g., Alpine vs. Debian). Choose based on the balance between size and your dependency requirements.

---
## 5. Docker Volumes

  

Docker Volumes provide persistent storage independent of the container’s lifecycle. They are used to store data such that it survives container removal and can be shared between containers.


### 5.1 Understanding Volume Mapping

  

- **Syntax:**  

  In a Docker run command or Compose file, volumes are specified in the format:

```text
  [host-path or volume-name]:[container-path]:[options]
```

- **Host Path vs. Named Volume:**  

  - **Host Path:** E.g., `d:/test:/app` mounts a local directory (`d:/test`) to a directory in the container (`/app`).  

  - **Named Volume:** E.g., `myvol:/app` uses a Docker-managed volume named `myvol`.  

    *Tip:*  

    - If you are not sure what container path to use, consult the documentation for your application or image. Often, a logical path such as `/app` or `/data` is used.

    - You can choose any valid absolute path inside the container—even if you don't know the pre-existing structure, you control where your files will be placed.

  

### 5.2 Common Volume Commands

  

- **Create a Volume:**
```bash
  docker volume create myvol
```

- **List Volumes:**
```bash
  docker volume ls
```

- **Inspect a Volume:**

```bash
  docker volume inspect myvol
```

- **Remove a Volume:**

```bash
  docker volume rm myvol
```

- **Prune Unused Volumes:**

```bash
  docker volume prune
```

  

### 5.3 Example Usage

  
```bash

# Create a Docker-managed volume and mount it into a container

docker volume create myvol

docker run -d --name devtest -v myvol:/app nginx:latest

# Mount a host directory as a bind mount (e.g., your local 'd:/test' directory)

docker run -d --name devtest -v d:/test:/app nginx:latest

```

  

_Explanation:_  

- For a **named volume** (`myvol:/app`), Docker manages the storage.  

- For a **bind mount** (`d:/test:/app`), you specify the exact path on your host system.  

- In both cases, the container will see the contents at the specified container path (here, `/app`).

  

---
## 6. Docker Compose Comprehensive Guide

**Docker Compose** allows you to define, run, and manage multi-container Docker applications using a single YAML file (usually named `docker-compose.yml`). In this guide, you’ll learn about all the essential commands and options—including networks, build contexts, volumes, secrets, environment variable substitution, restart policies, and more—to build robust, secure applications.

**YAML file structure and guide:**
- You can verify your YAML file [Here](https://www.yamllint.com/)

```yaml
# Comments in YAML look like this.

key: value
another_key: Another value goes here.
a_number_value: 100

# Nesting uses indentation. 2 space indent is preferred (but not required).
a_nested_map:
  key: value
  another_key: Another Value 
  another_nested_map:
    hello: hello

# Sequences (equivalent to lists or arrays) look like this
# (note that the ‘-‘ counts as indentation):
a_sequence:
  - Item 1
  - Item 2

# Since YAML is a superset of JSON, you can also write JSON-style maps and sequences:
json_map: {"key": "value"}
json_seq: [3, 2, 1, "takeoff"]

```

---

## 1. Basic Structure of a Docker Compose File

A typical Compose file includes:

- **version:** Specifies the file format version (e.g., `'3.8'`).
- **services:** Lists your containers and their configuration.
- **networks:** (Optional) Custom networks for inter-service communication.
- **volumes:** (Optional) Named volumes for persistent storage.
- **secrets:** (Optional) Securely pass sensitive data to services.
- **configs:** (Optional) Provide non-sensitive configuration files (mostly in Swarm mode).

Example skeleton:

```yaml
version: '3.8'

services:
  service1:
    image: some-image
    ports:
      - "80:80"
    environment:
      - ENV_VAR=value
    networks:
      - my-network

networks:
  my-network:
    driver: bridge

volumes:
  my-volume:

secrets:
  my_secret:
    file: ./secret.txt
```

---

## 2. Detailed Service Options and Commands

### 2.1 Build and Image Options

- **build:**  
    Build an image from a Dockerfile.
    
    ```yaml
    build:
      context: ./app
      dockerfile: Dockerfile.dev
    ```
    
    _Explanation:_
    
    - **context:** Directory containing the Dockerfile and necessary files.
    - **dockerfile:** Custom filename (if not using the default `Dockerfile`).
- **image:**  
    Use a pre-built image.
    
    ```yaml
    image: nginx:alpine
    ```
    
- **command:**  
    Override the container’s default command.
    
    ```yaml
    command: ["npm", "start"]
    ```
    

### 2.2 Port Mapping and Volumes

- **ports:**  
	    Map host ports to container ports.
    
    ```yaml
    ports:
      - "8080:80"
    ```
    
- **expose:**  
    Expose ports only to other containers (does not publish to host).
    
    ```yaml
    expose:
      - "3000"
    ```
    
- **volumes:**  
    Mount files or directories (host path or named volume).
    
    ```yaml
    volumes:
      - .:/app
      - data-volume:/data
    ```
    

### 2.3 Environment Variables and Variable Substitution

- **environment:**  
    Set environment variables inside the container.
    
    ```yaml
    environment:
      NODE_ENV: production
      PORT: 3000
    ```
    
    You can also use variable substitution:
    
    ```yaml
    environment:
      API_KEY: ${API_KEY}
    ```
    
    _Explanation:_ The variable `${API_KEY}` is substituted with the value from your shell or a `.env` file.
    
- **env_file:**  
    Load environment variables from a file.
    
    ```yaml
    env_file:
      - .env
    ```
    

### 2.4 Restart Policies and Health Checks

- **restart:**  
    Define how a container should restart.
    
    ```yaml
    restart: always
    ```
    
    _Options include:_ `no`, `always`, `on-failure`, `unless-stopped`.
    
- **healthcheck:**  
    Configure a command to monitor container health.
    
    ```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    ```
    

### 2.5 Dependencies, Extra Hosts, and Logging

- **depends_on:**  
    Define service startup order.
    
    ```yaml
    depends_on:
      - database
    ```
    
- **extra_hosts:**  
    Add custom host-to-IP mappings (like editing `/etc/hosts`).
    
    ```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ```
    
- **logging:**  
    Configure logging options.
    
    ```yaml
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    ```
    

### 2.6 Deploy (Swarm Mode Only)

- **deploy:**  
    Specify deployment settings (ignored in non-Swarm environments).
    
    ```yaml
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
      restart_policy:
        condition: on-failure
    ```
    

---

## 3. Using Secrets in Docker Compose

Docker Compose supports secrets to securely pass sensitive data. Secrets are typically stored in separate files and then mounted into containers as files in `/run/secrets/`.

### 3.1 Defining and Using Secrets

- **Defining a Secret:**
    
    ```yaml
    secrets:
      my_secret:
        file: ./secret.txt
    ```
    
    _Explanation:_ Docker Compose will load the contents of `secret.txt` as a secret named `my_secret`.
    
- **Using a Secret in a Service:**
    
    ```yaml
    services:
      app:
        image: my_app
        secrets:
          - source: my_secret
            target: my_custom_secret
    ```
    
    _Explanation:_
    
    - **source:** The secret defined in the secrets section.
    - **target:** The file name inside the container (default is the secret name if not specified).  
        In the container, the secret will be available at `/run/secrets/my_custom_secret`.

### 3.2 Environment Variable Substitution with Secrets

While Docker Compose secrets are mounted as files, you can use variable substitution in your Compose file to pass sensitive values from your environment:

```yaml
services:
  app:
    image: my_app
    environment:
      SECRET_KEY: ${MY_SECRET_KEY}
```

_Explanation:_ The value for `MY_SECRET_KEY` is taken from your environment or a `.env` file.

---

## 4. Networks, Volumes, and Configs

### 4.1 Custom Networks

- **Defining Networks:**
    
    ```yaml
    networks:
      frontend:
        driver: bridge
      backend:
        driver: bridge
    ```
    
- **Assigning Networks to Services:**
    
    ```yaml
    services:
      web:
        image: nginx:alpine
        networks:
          - frontend
      api:
        build: ./api
        networks:
          - backend
    ```
    

### 4.2 Named Volumes

- **Defining Named Volumes:**
    
    ```yaml
    volumes:
      data-volume:
    ```
    
- **Using Volumes in a Service:**
    
    ```yaml
    services:
      database:
        image: postgres:13
        volumes:
          - data-volume:/var/lib/postgresql/data
    ```
    

### 4.3 Configs (Swarm Mode and Advanced Use)

- **Defining Configs:**
    
    ```yaml
    configs:
      my_config:
        file: ./config.ini
    ```
    
- **Using Configs in a Service:**
    
    ```yaml
    services:
      app:
        image: my_app
        configs:
          - source: my_config
            target: /etc/config/config.ini
    ```
    

_Note:_ Configs are similar to secrets but are used for non-sensitive configuration data.

---

## 5. Detailed Docker Compose File Examples

### 5.1 Example 1: Simple Multi-Container Setup

```yaml
version: '3.8'
services:
  web:
    image: nginx:alpine
    container_name: webserver
    ports:
      - "8080:80"
    networks:
      - frontend

  app:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: my-app
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      PORT: 3000
      API_KEY: ${API_KEY}
    env_file:
      - .env
    volumes:
      - .:/app
    depends_on:
      - web
    restart: always
    networks:
      - frontend
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

volumes:
  data-volume:
```

_Explanation:_

- **web:** Uses a pre-built Nginx image, maps port 80 to 8080, and attaches to the `frontend` network.
- **app:** Builds from a Dockerfile, sets environment variables (including substitution from a `.env` file), mounts the current directory, and depends on the `web` service. It uses both `frontend` and `backend` networks.

### 5.2 Example 2: Advanced Setup with Secrets, Configs, and Custom Options

```yaml
version: '3.8'
services:
  database:
    image: postgres:13
    container_name: db
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: mydb
    volumes:
      - db-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - backend
    restart: unless-stopped

  backend:
    build:
      context: ./backend
    container_name: backend-app
    environment:
      - DATABASE_URL=postgres://myuser:mypassword@database:5432/mydb
    ports:
      - "5000:5000"
    depends_on:
      - database
    secrets:
      - my_secret
    networks:
      - backend
    restart: on-failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    container_name: frontend-app
    command: ["npm", "run", "serve"]
    ports:
      - "3000:3000"
    networks:
      - frontend
    depends_on:
      - backend
    restart: always
    extra_hosts:
      - "host.docker.internal:host-gateway"
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

secrets:
  my_secret:
    file: ./secret.txt

configs:
  my_config:
    file: ./config.ini

volumes:
  db-data:

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
```

_Explanation:_

- **database:** Runs PostgreSQL with a persistent volume (`db-data`), exposes port 5432, and uses a restart policy of `unless-stopped`.
- **backend:** Builds an image from the `./backend` directory, sets an environment variable for database connection, depends on the database, uses a secret (`my_secret`), includes a health check, and has an `on-failure` restart policy.
- **frontend:** Builds from `./frontend`, runs a custom command, maps ports, adds extra hosts, and configures logging.
- **Secrets and Configs:** Define a secret and a config file to securely pass sensitive and configuration data to services.

---

## 6.2 Common Docker Compose Commands

Use these commands to manage your multi-container application:

- **Build Images:**
    
    ```bash
    docker compose build
    ```
    
- **Start Services (detached):**
    
    ```bash
    docker compose up -d
    ```
    
- **Stop Services:**
    
    ```bash
    docker compose stop
    ```
    
- **Bring Down Services (remove containers, networks, volumes):**
    
    ```bash
    docker compose down
    ```
    
- **View Running Containers:**
    
    ```bash
    docker compose ps
    ```
    
- **View Logs for All Services:**
    
    ```bash
    docker compose logs
    # Follow logs in real time:
    docker compose logs --follow
    ```
    
- **Restart Services:**
    
    ```bash
    docker compose restart
    ```
    
- **Execute a Command Inside a Service Container:**
    
    ```bash
    docker compose exec <service_name> bash
    ```
    
- **Validate Your Compose File:**
    
    ```bash
    docker compose config
    ```
    
- **Copy Files Between Host and Container:**
    
    ```bash
    docker compose cp <service_name>:<source_path> <destination_path>
    docker compose cp <source_path> <service_name>:<destination_path>
    ```
    
- **Pull Latest Images for Services:**
    
    ```bash
    docker compose pull
    ```


---

**For any suggestions, feel free to contact on below Contact details:**

- Om Nagvekar Portfolio Website, Email: [Website](https://omnagvekar.github.io/) , [E-mail Address](mailto:omnagvekar29@gmail.com)
- GitHub, LinkedIn Profile:
    - Om Nagvekar: [GitHub](https://github.com/OmNagvekar)
    - Om Nagvekar: [LinkedIn](https://www.linkedin.com/in/om-nagvekar-aa0bb6228/)