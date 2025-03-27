import cv2
import numpy as np
import base64
import json
from datetime import datetime, timedelta

class WebcamQRCodeValidator:
    def __init__(self):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # QR code detector
        self.detector = cv2.QRCodeDetector()
    
    def validate_qr_data(self, qr_data):
        """
        Validate the QR code data structure and content
        
        Args:
            qr_data (str or dict): QR code data to validate
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # If qr_data is a string, try to parse it
            if isinstance(qr_data, str):
                qr_content = json.loads(qr_data)
            else:
                qr_content = qr_data
            
            # Required keys validation
            required_keys = {
                'keys': ['x0', 'y0', 'z0'],
                'location': ['lat', 'lng'],
                'timestamp': None,
                'expiration': None,
                'data': None
            }
            
            # Check for all required top-level keys
            for key in required_keys:
                if key not in qr_content:
                    return False, f"Missing required key: {key}"
            
            # Validate nested keys
            for key, subkeys in required_keys.items():
                if subkeys:
                    for subkey in subkeys:
                        if subkey not in qr_content[key]:
                            return False, f"Missing required subkey: {subkey} in {key}"
            
            # Validate timestamp and expiration
            try:
                timestamp = datetime.fromisoformat(qr_content['timestamp'])
                expiration = datetime.fromisoformat(qr_content['expiration'])
                
                # Check if QR code has expired
                current_time = datetime.now()
                if current_time > expiration:
                    return False, "QR code has expired"
                
                # Check time difference is within 5 minutes
                time_diff = expiration - timestamp
                if time_diff > timedelta(minutes=5):
                    return False, "Invalid time range"
                
            except ValueError:
                return False, "Invalid timestamp format"
            
            # Validate location
            lat = qr_content['location']['lat']
            lng = qr_content['location']['lng']
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                return False, "Invalid latitude or longitude"
            
            # Validate keys
            keys = qr_content['keys']
            for k in ['x0', 'y0', 'z0']:
                if not isinstance(keys[k], (int, float)):
                    return False, f"Invalid key type for {k}"
            
            # Validate base64 encoded data
            try:
                base64.b64decode(qr_content['data'])
            except Exception:
                return False, "Invalid base64 encoded data"
            
            return True, "QR code is valid"
        
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def create_rotation_overlay(self, frame, rotation_angle):
        """
        Create a frame with a rotation effect for invalid QR code
        
        Args:
            frame (numpy.ndarray): Original frame
            rotation_angle (float): Rotation angle in degrees
        
        Returns:
            numpy.ndarray: Rotated frame with overlay
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), rotation_angle, 1)
        
        # Perform rotation
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        
        # Create black overlay
        overlay = rotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        
        # Add "Invalid QR" text
        cv2.putText(overlay, 'INVALID QR', 
                    (width//2 - 200, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, rotated_frame, 0.3, 0, rotated_frame)
        
        return rotated_frame
    
    def scan_qr_code(self):
        """
        Scan QR code using webcam with immediate termination for invalid QR
        
        Returns:
            dict: Scanning and validation results
        """
        # Create a window for displaying result
        cv2.namedWindow('QR Code Validation', cv2.WINDOW_NORMAL)
        
        while True:
            # Read a frame from the webcam
            ret, frame = self.cap.read()
            
            if not ret:
                return {
                    'valid': False,
                    'error': 'Failed to capture frame'
                }
            
            # Detect and decode QR code
            data, bbox, _ = self.detector.detectAndDecode(frame)
            
            # Draw bounding box if QR code is detected
            if bbox is not None:
                # Convert bbox to integer for drawing
                bbox = bbox.astype(int)
                
                # Draw the bounding box
                cv2.polylines(frame, [bbox], True, (0, 255, 0), 3)
            
            # Check for QR code data
            if data:
                try:
                    # Validate the QR code data
                    is_valid, message = self.validate_qr_data(data)
                    
                    # If invalid, show rotation effect
                    if not is_valid:
                        # Rotation animation
                        for angle in range(0, 361, 30):
                            rotated_frame = self.create_rotation_overlay(frame, angle)
                            cv2.imshow('QR Code Validation', rotated_frame)
                            cv2.waitKey(50)  # Controls rotation speed
                        
                        # Release resources
                        self.cap.release()
                        cv2.destroyAllWindows()
                        
                        return {
                            'valid': False,
                            'message': message
                        }
                    
                    # If valid, proceed with further processing
                    parsed_data = json.loads(data)
                    
                    # Release the webcam and close windows
                    self.cap.release()
                    cv2.destroyAllWindows()
                    
                    return {
                        'valid': True,
                        'message': message,
                        'data': parsed_data
                    }
                
                except Exception as e:
                    print(f"Error processing QR code: {e}")
            
            # Display the frame
            cv2.imshow('QR Code Validation', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources if loop is exited
        self.cap.release()
        cv2.destroyAllWindows()
        
        return {
            'valid': False,
            'error': 'QR code scanning cancelled'
        }

def main():
    print("QR Code Scanner")
    print("Point your webcam at a QR code and press 'q' to quit")
    
    validator = WebcamQRCodeValidator()
    result = validator.scan_qr_code()
    
    print("\nQR Code Validation Result:")
    print(f"Valid: {result['valid']}")
    print(f"Message: {result.get('message', 'No message')}")
    
    if result['valid']:
        print("\nQR Code Details:")
        details = result['data']
        print(f"Location: {details['location']}")
        print(f"Timestamp: {details['timestamp']}")
        print(f"Expiration: {details['expiration']}")

if __name__ == "__main__":
    main()