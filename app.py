import logging
from flask import Flask, request, jsonify
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)



# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Connect to MySQL
def get_db_connection():
    engine = create_engine('mysql+mysqlconnector://root:@localhost/iprsr')
    return engine.connect()

# Fetch user's data based on userID from the MySQL database
def fetch_user_data(user_id):
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM user WHERE userID = '{user_id}'"
        user_data = pd.read_sql(query, conn)
        logging.debug(f"Fetched user data from user table for userID: {user_id}")
        return user_data
    finally:
        conn.close()

# Fetch parking data from the MySQL database
def fetch_parking_data(lot_id=None):
    """
    Fetch parking data from the database based on the provided lotID.
    If no lotID is specified, return an empty DataFrame.
    
    Parameters:
        lot_id (str): The lotID to filter parking data.

    Returns:
        pd.DataFrame: A DataFrame containing parking data.
    """
    try:
        conn = get_db_connection()

        if not lot_id:
            logging.error("No lotID specified.")
            return pd.DataFrame()  # Return an empty DataFrame if lotID is not provided

        # Fetch parking data for the specified lotID
        query = f"SELECT * FROM parkingspace WHERE lotID = '{lot_id}'"
        df = pd.read_sql(query, conn)
        logging.debug(f"Fetched {len(df)} parking spaces for lotID: {lot_id}")

        return df
    except Exception as e:
        logging.error(f"Error fetching parking data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_lots_coordinates():
    """
    Fetch all parking lots' lotID and coordinates from the parkinglot table and return as a dictionary.
    """
    try:
        conn = get_db_connection()
        query = "SELECT lotID, coordinates FROM parkinglot WHERE coordinates IS NOT NULL"  # Filter out rows with NULL coordinates
        df = pd.read_sql(query, conn)
        logging.debug(f"Fetched {len(df)} parking lots with lotID and coordinates from parkinglot table")

        # Convert the DataFrame to a dictionary
        lot_coordinates = {}
        for index, row in df.iterrows():
            lot_id = row['lotID']
            coordinates = row['coordinates']
            lat, lng = map(float, coordinates.split(','))  # Split and convert to float
            lot_coordinates[lot_id] = {'lat': lat, 'lng': lng}

        logging.debug(f"Location coordinates: {lot_coordinates}")
        return lot_coordinates
    except Exception as e:
        logging.error(f"Error fetching location coordinates from parkinglot table: {e}")
        return {}
    finally:
        conn.close()


def find_nearest_alternative_parking(current_lotID, k=3):
    """
    Find k-nearest alternative parking lots using KNN.
    Fetches coordinates dynamically from the parkinglot table.
    
    Parameters:
        current_location (str): The lotID of the current location.
        k (int): The number of nearest neighbors to find.
    
    Returns:
        list: A list of alternative lotIDs (excluding the current location).
    """
    try:
        # Fetch location coordinates from the parkinglot table
        lot_coordinates = get_lots_coordinates()

        # Check if the current location exists in the fetched coordinates
        if current_lotID not in lot_coordinates:
            logging.error(f"Current location '{current_lotID}' not found in parkinglot table.")
            return []

        # Convert location coordinates to a numpy array
        lots = list(lot_coordinates.keys())
        coordinates = np.array([[lot_coordinates[loc]['lat'], lot_coordinates[loc]['lng']] 
                              for loc in lots])
        
        # Initialize KNN
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(coordinates)
        
        # Get current location coordinates
        current_coords = np.array([[
            lot_coordinates[current_lotID]['lat'],
            lot_coordinates[current_lotID]['lng']
        ]])
        
        # Find k nearest neighbors
        distances, indices = knn.kneighbors(current_coords)
        
        # Return alternative locations (excluding the current location)
        alternative_locations = [lots[idx] for idx in indices[0] if lots[idx] != current_lotID]
        logging.debug(f"Found alternative locations: {alternative_locations}")
        return alternative_locations
        
    except Exception as e:
        logging.error(f"Error finding alternative parking lots: {str(e)}")
        return []

# Fetch user preferences from the MySQL database
def fetch_user_preferences(user_id):
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM parkingpreferences WHERE userID = '{user_id}'"
        
        # Fetch user preferences using the query
        user_prefs = pd.read_sql(query, conn)
        logging.debug(f"Fetched user preferences for userID: {user_id}")
        
        # Drop 'preferencesID' and 'userID' columns to only keep relevant preferences
        user_prefs_filtered = user_prefs.drop(columns=['preferencesID', 'userID'])
        
        logging.debug(f"Fetched and filtered user preferences for userID: {user_id}")
        
        # Convert the DataFrame to a dictionary of scalar values
        user_preferences_dict = user_prefs_filtered.iloc[0].to_dict()
        
        logging.debug(f"User preferences as dictionary: {user_preferences_dict}")
        
        return user_preferences_dict  # Return as a dictionary of scalars
    finally:
        conn.close()

# Mapping between user preferences and the column names in the parking table
preference_to_parking_col_map = {
    'isNearest': 'isNearest',
    'isCovered': 'isCovered',
    'requiresLargeSpace': 'hasLargeSpace',
    'requiresWellLitArea': 'isWellLitArea',
    'requiresEVCharging': 'hasEVCharging',
    'requiresWheelchairAccess': 'isWheelchairAccessible',
    'requiresFamilyParkingArea': 'isFamilyParkingArea',
    'premiumParking': 'isPremium'
}


def calculate_similarity(row, preferences, weights):
    score = 0.0  # Start with a float score to avoid integer division issues

    # Iterate through user preferences and map them to parking column names
    for user_pref, pref_value in preferences.items():
        parking_col = preference_to_parking_col_map.get(user_pref)  # Get corresponding column name
        if parking_col in row and parking_col in weights:  # Ensure the attribute exists in both row and weights
            if pd.notnull(row[parking_col]) and pd.notnull(pref_value):  # Ensure both row and pref_value are not null
                if pd.api.types.is_scalar(row[parking_col]) and row[parking_col] == pref_value:  # Ensure scalar comparison
                    logging.debug(f"Processing attribute: {parking_col} with user preference: {pref_value} and parking value: {row[parking_col]}")
                    score += weights[parking_col]  # Add the weight to the score if there's a match
                    logging.debug(f"Attribute match found for {parking_col}: Adding weight {weights[parking_col]} to score.")
                else:
                    logging.debug(f"No match for {parking_col}. User preference: {pref_value}, Parking value: {row[parking_col]}")
        else:
            logging.debug(f"Skipping attribute {user_pref}, corresponding parking column {parking_col} not found.")
    
    return float(score) 



# Suggest parking logic
def recommend_parking(df, user_preferences, is_female, has_disability):
    logging.debug(f"Starting recommendation process for user preferences: {user_preferences}")

    attribute_weights = {
        'isNearest': 1,
        'isCovered': 0.5,
        'hasLargeSpace': 0.7,
        'hasEVCharging': 0.6,
        'isWheelchairAccessible': 1.5,
        'isWellLitArea': 0.4,
        'isFamilyParkingArea': 0.5,
        'isPremium': 0.5
    }

    df_available = df[df['isAvailable'] == 1].copy()
    logging.debug(f"Found {len(df_available)} available parking spaces")

    # Step 1: Premium parking (moved to top priority)
    if user_preferences.get('premiumParking', 0) == 1:
        logging.debug("User requires premium parking, searching for premium parking...")
        df_premium = df_available[(df_available['parkingType'] == 'Premium') & (df_available['isPremium'] == 1)]
        if not df_premium.empty:
            logging.debug(f"Found {len(df_premium)} premium parking spaces")
            df_premium['similarity'] = df_premium.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
            best_premium_space = df_premium.sort_values(by='similarity', ascending=False).iloc[0]
            return best_premium_space

    # Step 2: Special parking for users with disabilities
    if has_disability:
        logging.debug("User has a disability, searching for special parking...")
        df_special = df_available[(df_available['parkingType'] == 'Special') & (df_available['isWheelchairAccessible'] == 1)]
        if not df_special.empty:
            logging.debug(f"Found {len(df_special)} special parking spaces for disability")
            df_special['similarity'] = df_special.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
            best_special_space = df_special.sort_values(by='similarity', ascending=False).iloc[0]
            return best_special_space

    # Step 3: Parking for female users in well-lit areas
    if is_female:
        logging.debug("User is female, searching for female parking in well-lit areas...")
        df_female = df_available[(df_available['parkingType'] == 'Female') & (df_available['isWellLitArea'] == 1)]
        if not df_female.empty:
            logging.debug(f"Found {len(df_female)} female parking spaces in well-lit areas")
            df_female['similarity'] = df_female.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
            best_female_space = df_female.sort_values(by='similarity', ascending=False).iloc[0]
            return best_female_space

    # Step 4: Parking with EV Charging for users with EV cars
    if user_preferences.get('requiresEVCharging', 0) == 1:
        logging.debug("User requires EV charging, searching for EV parking...")
        df_ev = df_available[(df_available['parkingType'] == 'EV Car') & (df_available['hasEVCharging'] == 1)]
        if not df_ev.empty:
            logging.debug(f"Found {len(df_ev)} EV charging parking spaces")
            df_ev['similarity'] = df_ev.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
            best_ev_space = df_ev.sort_values(by='similarity', ascending=False).iloc[0]
            return best_ev_space

    # Step 5: Parking for users with families
    if user_preferences.get('requiresFamilyParkingArea', 0) == 1:
        logging.debug("User requires family parking, searching for family parking...")
        df_family = df_available[(df_available['parkingType'] == 'Family') & (df_available['isFamilyParkingArea'] == 1)]
        if not df_family.empty:
            logging.debug(f"Found {len(df_family)} family parking spaces")
            df_family['similarity'] = df_family.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
            best_family_space = df_family.sort_values(by='similarity', ascending=False).iloc[0]
            return best_family_space

    # Step 6: Fallback to Regular parking and calculate the best match based on similarity
    logging.debug("No specialized parking found, falling back to regular parking...")
    df_regular = df_available[df_available['parkingType'] == 'Regular']
    if not df_regular.empty:
        logging.debug(f"Found {len(df_regular)} regular parking spaces")
        df_regular['similarity'] = df_regular.apply(lambda row: calculate_similarity(row, user_preferences, attribute_weights), axis=1)
        df_regular = df_regular[df_regular['similarity'] > 0]
        if not df_regular.empty:
            best_regular_space = df_regular.sort_values(by='similarity', ascending=False).iloc[0]
            return best_regular_space

    # If no suitable parking space is found
    logging.debug("No suitable parking space found")
    return None


# Route for suggesting parking based on userID
@app.route('/recommend-parking', methods=['POST'])
def recommend_parking_endpoint():
    data = request.json
    user_id = data.get('userID')
    lot_id = data.get('lotID')  # Change from 'location' to 'lotID'

    if not user_id or not lot_id:
        logging.error("userID and lotID are required")
        return jsonify({"status": "error", "message": "userID and lotID are required"}), 400

    try:
        # Fetch user preferences from the database using userID
        user_preferences = fetch_user_preferences(user_id)
        
        # Check if user preferences are empty
        if not user_preferences:
            logging.debug(f"No user preferences found for userID: {user_id}")
            return jsonify({"status": "error", "message": "User preferences not found"}), 404

        # Fetch the parking data dynamically from the database based on lotID
        parking_df = fetch_parking_data(lot_id)

        # Determine if the user is female and/or has a disability from preferences
        user_data = fetch_user_data(user_id)
        is_female = user_data['gender'].values[0] == 1
        has_disability = user_data['hasDisability'].values[0] == 1

        logging.debug(f"User isFemale: {is_female}, hasDisability: {has_disability}")

        # Call recommendation logic
        recommended_parking = recommend_parking(parking_df, user_preferences, is_female, has_disability)

        if recommended_parking is not None:
            parking_id = recommended_parking['parkingSpaceID']
            logging.debug(f"Returning recommended parking space: {parking_id}")
            return jsonify({"status": "success", "parkingSpaceID": parking_id})
        
        logging.debug("No recommendations found in preferred location")
        
        # Look for alternatives
        alternative_locations = find_nearest_alternative_parking(lot_id)  # Use lotID here
        
        for alt_location in alternative_locations:
            # Fetch parking data for alternative location
            alt_parking_df = fetch_parking_data(alt_location)
            alt_recommended_parking = recommend_parking(alt_parking_df, user_preferences, is_female, has_disability)
            
            if alt_recommended_parking is not None:
                parking_id = alt_recommended_parking['parkingSpaceID']
                logging.debug(f"Found alternative parking in {alt_location}: {parking_id}")
                return jsonify({
                    "status": "success",
                    "parkingSpaceID": parking_id,
                    "message": f"Original location full. Found parking at {alt_location}",
                    "alternativeLocation": alt_location
                })
        
        return jsonify({
            "status": "info", 
            "message": "No parking spaces available in nearby locations"
        }), 200

    except Exception as e:
        logging.error(f"Error during recommendation process: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)