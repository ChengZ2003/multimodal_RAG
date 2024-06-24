# multimodal_RAG

## Get Started

Follow these steps to get started with this project:

### 1. Install Dependencies

First, install all the necessary Python dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

Next, start Milvus by navigating to the dependencies directory and running Docker Compose:

```bash
cd dependencies/milvus
docker-compose up -d
```

### 3. Start Nebula Graph

Finally, start Nebula Graph by navigating to the corresponding directory and running the installation script:

```bash
cd dependencies/nebulaGraph
bash install.sh
```
