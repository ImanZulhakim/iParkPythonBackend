# iPARK Backend - Intelligent Parking Recommendation System

This repository contains the backend implementation of the iPARK system, which powers personalized parking recommendations and efficient parking management. It is built using Python (Flask) and integrates seamlessly with the MySQL database and other components of the iPARK ecosystem.

## Key Features

- **User Data Management**: Fetch user data and preferences to provide tailored recommendations.
- **Parking Data Management**: Retrieve real-time parking availability and attributes from the database.
- **Personalized Recommendations**: Utilize K-Nearest Neighbors (KNN) for dynamic, user-specific parking recommendations.
- **Alternative Parking Suggestions**: Suggest nearby parking lots when the primary location is full.
- **Special Needs Handling**: Prioritize parking spaces for users with disabilities, EV charging requirements, and family-friendly options.
- **Logging and Error Handling**: Comprehensive logging to ensure smooth operations and troubleshooting.

## Technologies Used

- **Flask**: Lightweight Python web framework.
- **MySQL**: Relational database for storing user, parking, and preference data.
- **SQLAlchemy**: Database ORM for efficient query execution.
- **scikit-learn**: Used for implementing KNN recommendation logic.
- **Pandas**: Data analysis and manipulation.
- **NumPy**: Efficient numerical computations.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- MySQL server
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ipark-backend.git
   cd ipark-backend
2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
4. Configure the MySQL database:
   - Import the provided database schema (schema.sql) into your MySQL server.
   - Update the database connection string in the get_db_connection() function in the app.py file:
python

   ```bash
   pip install -r requirements.txt
6. Start the Flask server:
   ```bash
   python app.py


## Related Repositories
- **[iPARK Android Application](https://github.com/username/iPark-Mobile-App):** iPark Mobile App  
- **[iPARK Web-Based Admin Panel](https://github.com/username/iPark-Web-Admin-Panel):** iPark Web Admin Panel


## Contributing
Contributions are welcome! If you have an idea or find an issue, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
