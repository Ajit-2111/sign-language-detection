# Sign Language Detection with Flask

Welcome to the Sign Language Detection project! This Python Flask application uses a machine learning model to detect American Sign Language (ASL) gestures in real-time from a video feed. Users can easily integrate this project into their applications or use it as a standalone tool for ASL interpretation.

## Features

- Real-time American Sign Language detection from a video stream.
- Flask web application for easy deployment and integration.
- Simple and intuitive user interface for interacting with the model.


## Getting Started

Follow these instructions to set up and run the Sign Language Detection project on your local machine.

### Prerequisites

- Python 3.9 or higher
- Clone this repository:

```bash
git clone https://github.com/Ajit-2111/sign-language-detection.git
cd sign-language-detection
```

### Running the Application

1. Install required packages using:

```bash
pip install -r requirements.txt
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

4. Use your device's camera to capture ASL gestures, and the model will display the corresponding ASL value in real-time.



## Project Structure

- **app.py**: Flask application script containing the main server logic.
- **static**: Directory for static files such as CSS and JavaScript.
- **templates**: HTML templates for rendering the web pages.
- **requirements.txt**: List of Python packages required for the project.

## Model

The ASL detection model used in this project is trained on a custom dataset of hand gestures representing American Sign Language. 

## Contributing

Contributions are welcome! If you have ideas for improvements, bug reports, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Enjoy using Sign Language Detection with Flask! If you have any questions or feedback, feel free to reach out.

Happy signing! 🤟
