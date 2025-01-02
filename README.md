# hands-on_machine_learning_oreilly

Exercises and examples from O'Reilly's "Hands-on Machine Learning with Scikit-Learn, Keras &amp; TernsorFlow" by A. Geron (3rd ed.)

## Description

This repository is an accompanying journal on my path of going through examples and exercises from ["Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"](https://www.amazon.ca/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975/ref=sr_1_1?crid=38N00QAG38IJI&dib=eyJ2IjoiMSJ9.0Jm1bLoXUBA_cK-tar7HW1L7KzUCRFLa7mcRAFq-rLTjFvovU0-vJBQMzTURv4kIz7873GuN38RCcSP06ZFwPEAr4narzSgskAqqh4m2BCdfvOvHjEXN6nHdT6SCnDxb8dpaG1ry6YSeG1ez78MzlOgf3Nhf-Q1YNPn5JF_1laMUQtr6BI-sSz9SaIAii-dNKbtBEl9xtq7CrgwaI2iaiI3edXso3KcduUeBa3uAG7xnh0XiKAgLk_ClZUQYwly1icOOvbZ-dvlK4c88g2NAafKIWWqXja0xyho3m8o86fw.ZVOwAPQxI9PfqNFYoW5k53GPy89mH6wLPat9B7v2tos&dib_tag=se&keywords=hands-on+machine+learning+with+scikit-learn%2C+keras%2C+and+tensorflow&qid=1716585380&sprefix=hands-on+mac%2Caps%2C66&sr=8-1) book from o'Reilly.
It follows book's chapter division, partially for ease of navigation, but also due to the fact that different chapters
may require different environment set-up.

## Runbook

### For Python modules

Please follow these steps to set-up and activate a local virtual environment:

1. Make sure you're at the root of the project
2. Set-up the virtual environment

   ```shell
   python -m venv shared_venv/
   ```

3. Activate your virtual environment (example below is for macOS/Linux)

   ```shell
   source shared_venv/bin/activate
   ```

4. Install required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

5. [optional, for Visual Studio Code] If you want to run/edit the project from Visual Studio Code:
   1. Make sure you have _Python extension for Visual Studio Code_ installed
   2. Open command pallete (`Ctr + Shift + P` on Linux) and select _Python: Select Interpreter_
   3. Pick the option containing `shared_venv`

To run a Python file, run a `main` module like this:

```shell
python3 -m chapter_03.ex03.main
```

To run tests for a certain chapter:

```shell
# change chapter folder accordingly
cd chapter_02/
python -m pytest
```

### Docker setup

For Jupyter Notebook examples I've used [initial containerized environment](https://github.com/ageron/handson-ml3/tree/main/docker) with a [small fix](https://github.com/ageron/handson-ml3/pull/144). Simply navigate to `docker` folder and just run: `docker compose up`.

This setup is based on [Dockerfile.gpu](docker/Dockerfile.gpu).

#### Build image

This should be ideally done only once:

```shell
cd docker/
# build docker compose image from Dockerfile.gpu
# Note that this may take a while (12 minutes on my machine)
docker compose build
```

#### Run

```shell
# click on one of the produced links to start adding/editing Jupyter Notebook entries
docker compose up
...
handson-ml3  |     To access the server, open this file in a browser:
handson-ml3  |         file:///home/devel/.local/share/jupyter/runtime/jpserver-1-open.html
handson-ml3  |     Or copy and paste one of these URLs:
handson-ml3  |         http://62903a64ecf6:8888/lab?token=87cbe0058148e33d6702d94d243525acb18f7577ff00ca5d
handson-ml3  |         http://127.0.0.1:8888/lab?token=87cbe0058148e33d6702d94d243525acb18f7577ff00ca5d
handson-ml3  | [I 2024-06-24 23:49:32.937 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
```

⚠️ Should the above docker container be automatically restarted by the Docker engine on your laptop, just look for the last logs for the running container, e.g.:

```shell
docker logs --tail 15 handson-ml3
    To access the server, open this file in a browser:
        file:///home/devel/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://62903a64ecf6:8888/lab?token=b99d270fee1d46985f3f58abb29dabf913672ff971da673c
        http://127.0.0.1:8888/lab?token=b99d270fee1d46985f3f58abb29dabf913672ff971da673c
[I 2024-08-19 20:10:40.091 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
[I 2024-08-20 22:19:51.983 ServerApp] 302 GET / (@172.18.0.1) 0.41ms
[I 2024-08-20 22:19:51.985 LabApp] 302 GET /lab? (@172.18.0.1) 0.46ms
[I 2024-08-20 22:20:23.947 LabApp] 302 GET /lab?token=[secret] (@172.18.0.1) 0.52ms
[W 2024-08-20 22:22:28.435 LabApp] Could not determine jupyterlab build status without nodejs
[I 2024-08-20 22:22:29.358 ServerApp] Kernel started: b6936699-0de1-4051-a6a1-337ea2696f9c
[I 2024-08-20 22:22:30.748 ServerApp] Connecting to kernel b6936699-0de1-4051-a6a1-337ea2696f9c.
[I 2024-08-20 22:22:30.762 ServerApp] Connecting to kernel b6936699-0de1-4051-a6a1-337ea2696f9c.
[I 2024-08-20 22:22:30.776 ServerApp] Connecting to kernel b6936699-0de1-4051-a6a1-337ea2696f9c.
```
