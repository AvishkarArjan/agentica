# Agentica Deep Research Agent

A deep research assistant designed to help users discover and analyze information effortlessly. It leverages intelligent tools and a streamlined interface to provide insightful, real-time research capabilities.

## Features

- 🔍 Deep research and document analysis  
- ⚡ Fast and responsive UI  
- 🎨 Customizable UI themes using Gradio

## 🚀 Setup & Configuration Guide

#### 📦 Part 1: Project Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/AvishkarArjan/agentica
   ```
   *or*  
   [Download ZIP](https://github.com/AvishkarArjan/agentica) and extract it manually.

2. **Navigate into the Project Directory**  
   ```bash
   cd agentica
   ```

3. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. **Install Required Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

#### 🔑 Part 2: Setting Up API Keys

#### Hugging Face
- Go to [Hugging Face Login](https://huggingface.co/login)
- After logging in:
  - Click on your profile (top right)
  - Go to **Access Tokens**
  - Click **"New Token"** to generate one

#### Tavily
- Log in at [https://tavily.com/](https://tavily.com/)
- Navigate to your dashboard and **Generate New API Key**

---

#### ⚙️ Part 3: Configure Environment Variables

- In the project directory, navigate to the config folder:
  ```bash
  cd config
  ```
- Open the `.env` file and add your API keys:
  ```env
  HUGGINGFACE_API_KEY=your_huggingface_token
  TAVILY_API_KEY=your_tavily_key
  ```


## Resources

- **UI Themes Guide**: [Gradio Theming Guide](https://www.gradio.app/guides/theming-guide)

