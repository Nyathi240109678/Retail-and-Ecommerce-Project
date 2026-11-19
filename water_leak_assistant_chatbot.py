
# WATER LEAK ASSISTANT CHATBOT
# A chatbot that provides water leak guidance,
# reads detection alerts and prepares maintenance reports.
# Uses CountVectorizer and Multinomial Naive Bayes.

import os
import csv
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1. FILE LOCATION

folder = os.path.dirname(os.path.abspath(__file__))

report_path = os.path.join(folder, "maintenance_alerts.csv")
detection_path = os.path.join(folder, "leak_alerts_output.csv")


# 2. CHATBOT TRAINING DATA

training_data = {

    "leak_detected": [
        "a leak has been detected",
        "what should i do if a leak is detected",
        "the system detected a leak",
        "there is a water leak",
        "what do i do about a detected leak",
        "how should we respond to a leak",
        "the network has a leakage",
        "what happens when a leak is detected",
        "how do i respond to a water leak",
        "there may be a leak in the network"
    ],

    "maintenance_alert": [
        "alert the maintenance team",
        "prepare a maintenance report",
        "notify maintenance about the leak",
        "create a maintenance alert",
        "i want to report a leak",
        "generate a leak report",
        "prepare an incident report",
        "how do i inform the maintenance team",
        "record this leak",
        "make a maintenance notification"
    ],

    "pressure_issue": [
        "why is water pressure dropping",
        "what causes low water pressure",
        "the pressure is too low",
        "why is the pressure changing",
        "what does unusual pressure mean",
        "how does pressure help detect leaks",
        "what happens when pressure decreases",
        "explain pressure readings",
        "why is the pressure unstable",
        "what is a pressure sensor"
    ],

    "flow_issue": [
        "what does unusual water flow mean",
        "why is the water flow increasing",
        "what causes abnormal flow readings",
        "explain flow measurements",
        "what is a flow sensor",
        "why is water flowing when it should not",
        "how can flow readings detect leaks",
        "what happens when flow changes",
        "why is the flow reading high",
        "explain water flow monitoring"
    ],

    "normal_status": [
        "is the network operating normally",
        "is there a leak",
        "what does normal status mean",
        "is the water network safe",
        "has the leak stopped",
        "what does no leak detected mean",
        "is the system detecting a problem",
        "what is the current network status",
        "is everything normal",
        "explain the normal detection status"
    ],

    "leak_prevention": [
        "how can we prevent water leaks",
        "how do we prevent pipe leaks",
        "ways to prevent water leakage",
        "how can leaks be avoided",
        "how do i maintain water pipes",
        "how can we protect water pipes",
        "how can pipe damage be prevented",
        "how do we reduce pipe failures",
        "give me leak prevention advice",
        "how can water infrastructure be maintained"
    ],

    "water_conservation": [
        "how can we save water",
        "how can we reduce water wastage",
        "why is water conservation important",
        "how do leaks waste water",
        "how can a water utility conserve water",
        "ways to reduce water losses",
        "how can we use water efficiently",
        "how do water leaks affect the environment",
        "give me water saving tips",
        "how can we reduce treated water losses"
    ],

    "emergency": [
        "what should i do during a major leak",
        "there is a burst pipe",
        "what do i do in a water emergency",
        "there is flooding from a pipe",
        "how do i respond to a burst water pipe",
        "what should happen during a serious leak",
        "there is water flooding the area",
        "who should handle a major leak",
        "what is the emergency procedure",
        "how do i report a burst pipe"
    ],

    "how_it_works": [
        "how does the leak detection system work",
        "explain the water leak model",
        "how does machine learning detect leaks",
        "what is the purpose of this project",
        "how does the chatbot work",
        "what data does the model use",
        "how are leaks identified",
        "explain the detection process",
        "what algorithm is used",
        "how does the system monitor the network"
    ],

    "sensors": [
        "what sensors does the system use",
        "what is a pressure sensor",
        "what is a flow sensor",
        "what does the tank level sensor measure",
        "why are sensors important",
        "how do sensors monitor water networks",
        "what readings are collected",
        "explain the network sensors",
        "what is a water demand sensor",
        "how are sensor readings used"
    ]
}


training_questions = []
training_labels = []

for intent_name, questions in training_data.items():
    for question in questions:
        training_questions.append(question)
        training_labels.append(intent_name)


# 3. TRAIN THE CHATBOT

vectorizer = CountVectorizer()
X_chat = vectorizer.fit_transform(training_questions)

chat_model = MultinomialNB()
chat_model.fit(X_chat, training_labels)


# 4. CHATBOT RESPONSES

responses = {

    "leak_detected": (
        "LEAK DETECTION GUIDANCE\n"
        "----------------------------------------\n"
        "If a significant leak is detected, the following actions "
        "are recommended:\n\n"
        "1. Notify the responsible water network operator.\n"
        "2. Review the pressure, flow and tank-level readings.\n"
        "3. Investigate the affected network section using available "
        "monitoring information.\n"
        "4. Arrange for the maintenance team to inspect the suspected "
        "area.\n"
        "5. If required, an authorised operator should isolate the "
        "appropriate section according to the utility's procedures.\n\n"
        "IMPORTANT: The detection model identifies a possible leak "
        "at network level. It does not identify the exact leaking pipe."
    ),

    "pressure_issue": (
        "WATER PRESSURE GUIDANCE\n"
        "----------------------------------------\n"
        "- A sudden pressure decrease may indicate a possible leak.\n"
        "- Pressure changes can also result from high water demand.\n"
        "- Pump operation may affect pressure readings.\n"
        "- Compare readings from different pressure sensors.\n"
        "- Investigate unusual or persistent pressure changes.\n\n"
        "Pressure readings should be assessed together with flow "
        "and tank-level information."
    ),

    "flow_issue": (
        "WATER FLOW GUIDANCE\n"
        "----------------------------------------\n"
        "- Unusual flow increases may indicate water loss.\n"
        "- Unexpected flow changes should be investigated.\n"
        "- Compare flow readings with normal operating patterns.\n"
        "- Consider pump activity and changes in water demand.\n"
        "- Report abnormal readings to the network operator."
    ),

    "normal_status": (
        "NORMAL NETWORK STATUS\n"
        "----------------------------------------\n"
        "A normal status means the detection system is not currently "
        "reporting a significant leak.\n\n"
        "Continue monitoring the network regularly.\n"
        "Remember that a normal prediction does not guarantee that "
        "there are no small leaks."
    ),

    "leak_prevention": (
        "WATER LEAK PREVENTION\n"
        "----------------------------------------\n"
        "- Inspect pipes and connections regularly.\n"
        "- Monitor pressure levels across the network.\n"
        "- Maintain pumps, valves and other equipment.\n"
        "- Investigate unusual flow and pressure readings.\n"
        "- Repair damaged infrastructure promptly.\n"
        "- Keep accurate maintenance records."
    ),

    "water_conservation": (
        "WATER CONSERVATION ADVICE\n"
        "----------------------------------------\n"
        "- Detect and repair leaks as early as possible.\n"
        "- Monitor water losses across the network.\n"
        "- Maintain pipes and distribution equipment.\n"
        "- Avoid unnecessary water wastage.\n"
        "- Monitor water consumption patterns.\n"
        "- Prioritise maintenance in areas with repeated problems.\n\n"
        "Early leak detection helps reduce the loss of treated water."
    ),

    "emergency": (
        "WATER LEAK EMERGENCY GUIDANCE\n"
        "----------------------------------------\n"
        "If a major leak or burst pipe occurs:\n\n"
        "1. Keep people away from flooded or potentially dangerous areas.\n"
        "2. Contact the responsible water utility or emergency team.\n"
        "3. Provide the location and details of the incident.\n"
        "4. Arrange an urgent inspection by qualified personnel.\n"
        "5. An authorised operator should isolate the affected section "
        "if it is safe and appropriate to do so.\n\n"
        "Do not attempt repairs on pressurised water infrastructure."
    ),

    "how_it_works": (
        "HOW THE WATER LEAK DETECTION SYSTEM WORKS\n"
        "----------------------------------------\n"
        "1. The system receives historical water network readings.\n"
        "2. It analyses pressure, flow and tank-level measurements.\n"
        "3. Machine learning identifies patterns associated with "
        "significant leaks.\n"
        "4. The model estimates the probability of a leak.\n"
        "5. An alarm is raised when the detection conditions are met.\n\n"
        "The project uses a machine learning classification model "
        "to identify significant network-level leak conditions."
    ),

    "sensors": (
        "WATER NETWORK SENSORS\n"
        "----------------------------------------\n"
        "- Pressure sensors measure water pressure at network nodes.\n"
        "- Flow sensors measure the movement of water through pipes "
        "and pumps.\n"
        "- Tank-level sensors monitor the amount of water in storage.\n"
        "- Demand measurements represent water consumption across "
        "the network.\n\n"
        "The system analyses available sensor readings to identify "
        "patterns that may indicate a significant leak."
    )
}


greeting_response = (
    "WATER LEAK ASSISTANT\n"
    "----------------------------------------\n"
    "Hello! Welcome to the Water Leak Assistant.\n\n"
    "I can help you with:\n"
    "- Water leak detection\n"
    "- Maintenance alerts\n"
    "- Pressure and flow problems\n"
    "- Leak prevention\n"
    "- Water conservation\n"
    "- Emergency guidance\n"
    "- How the detection system works"
)

thanks_response = (
    "WATER LEAK ASSISTANT\n"
    "----------------------------------------\n"
    "You're welcome! I am here to assist with water leak "
    "detection and network maintenance questions."
)

unrelated_response = (
    "WATER LEAK ASSISTANT\n"
    "----------------------------------------\n"
    "I'm not sure what you mean.\n"
    "I can assist with water leak detection, maintenance, pressure, "
    "flow, emergencies and water conservation.\n\n"
    "Try asking:\n"
    "- What should I do if a leak is detected?\n"
    "- Alert the maintenance team.\n"
    "- Why is water pressure dropping?\n"
    "- How can we prevent water leaks?\n"
    "- How does the detection system work?"
)


# 5. DETECTION INFORMATION

latest_detection = {
    "status": "UNKNOWN",
    "probability": None,
    "alarm_status": "UNKNOWN",
    "timestamp": None,
    "details": "No detection information has been provided."
}


def update_detection_status(
    status,
    probability=None,
    alarm_status=None,
    timestamp=None,
    details=None
):

    global latest_detection

    status = str(status).strip().upper()

    if status not in ("LEAK", "NORMAL", "UNKNOWN"):
        raise ValueError("Status must be LEAK, NORMAL or UNKNOWN.")

    if probability is not None:
        probability = float(probability)

        if probability < 0 or probability > 100:
            raise ValueError("Probability must be between 0 and 100.")

    latest_detection = {
        "status": status,
        "probability": probability,
        "alarm_status": (
            str(alarm_status).strip().upper()
            if alarm_status is not None
            else "UNKNOWN"
        ),
        "timestamp": (
            str(timestamp)
            if timestamp is not None
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "details": (
            str(details)
            if details
            else "Network-level detection information."
        )
    }


# 6. READ LATEST DETECTION FROM CSV

def load_detection_status():

    if not os.path.exists(detection_path):
        update_detection_status(
            status="UNKNOWN",
            details="The detection output file could not be found."
        )
        return

    try:
        with open(
            detection_path,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)
            rows = list(reader)

            if not rows:
                update_detection_status(
                    status="UNKNOWN",
                    details="The detection output file is empty."
                )
                return

            latest_row = rows[-1]

            probability = float(latest_row["leak_probability"])
            reading = str(latest_row["reading"]).strip().upper()
            alarm = str(latest_row["alarm"]).strip().upper()

            if probability <= 1:
                probability *= 100

            if alarm == "ON":
                status = "LEAK"
            elif alarm in ("OFF", "FALSE", "0"):
                status = "NORMAL"
            else:
                status = "UNKNOWN"

            timestamp = (
                latest_row.get("timestamp")
                or latest_row.get("Timestamp")
                or latest_row.get("Unnamed: 0")
                or latest_row.get("")
                or "Not available"
            )

            update_detection_status(
                status=status,
                probability=round(probability, 2),
                alarm_status=alarm,
                timestamp=timestamp,
                details=(
                    f"Individual sensor prediction: {reading}. "
                    f"Confirmed network alarm: {alarm}."
                )
            )

    except (ValueError, KeyError, TypeError) as error:

        update_detection_status(
            status="UNKNOWN",
            details=f"Unable to read detection information: {error}"
        )

    except OSError as error:

        update_detection_status(
            status="UNKNOWN",
            details=f"Unable to open detection file: {error}"
        )


# 7. DISPLAY DETECTION STATUS

def show_detection_status():

    load_detection_status()

    print("\n" + "=" * 55)
    print("              LATEST DETECTION STATUS")
    print("=" * 55)

    print("Detection status:", latest_detection["status"])
    print("Alarm status:", latest_detection["alarm_status"])

    if latest_detection["probability"] is not None:
        print(
            "Leak probability:",
            str(latest_detection["probability"]) + "%"
        )

    print("Date and time:", latest_detection["timestamp"])
    print("Details:", latest_detection["details"])

    if latest_detection["status"] == "LEAK":
        print("\nWARNING: A possible significant network leak was detected.")
        print("Recommended action: Arrange a maintenance investigation.")

    elif latest_detection["status"] == "NORMAL":
        print("\nNo significant leak is currently being reported.")

    else:
        print("\nNo detection information is currently available.")

    print("=" * 55)


# 8. MAINTENANCE REPORT

def create_maintenance_report():

    load_detection_status()

    print("\n" + "=" * 55)
    print("              MAINTENANCE ALERT")
    print("=" * 55)

    print("Preparing a maintenance incident report...")

    if latest_detection["status"] == "UNKNOWN":
        print("\nNo detection information is available.")
        print("Please provide detection information first.")
        return

    if latest_detection["status"] == "NORMAL":
        print("\nThe latest detection status is NORMAL.")
        print("A leak maintenance alert will not be created.")
        return

    print("\nA possible significant network leak has been reported.")
    print("The report will contain the available detection information.")

    location = input(
        "\nEnter the known network area or location "
        "(or press Enter if unknown): "
    ).strip()

    if not location:
        location = "Not specified"

    description = input(
        "Enter additional incident details "
        "(or press Enter to skip): "
    ).strip()

    if not description:
        description = "No additional details provided."

    report = {
        "Incident_ID": datetime.now().strftime("LEAK%Y%m%d%H%M%S"),
        "Date_Time": latest_detection["timestamp"],
        "Detection_Status": latest_detection["status"],
        "Leak_Probability": latest_detection["probability"],
        "Alarm_Status": latest_detection["alarm_status"],
        "Network_Area": location,
        "Incident_Details": description,
        "Recommended_Action": (
            "Notify the network operator and arrange an inspection "
            "of the suspected network section."
        ),
        "Report_Status": "Prepared - Not Sent"
    }

    file_exists = os.path.exists(report_path)

    with open(report_path, "a", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=report.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(report)

    print("\n" + "=" * 55)
    print("        MAINTENANCE REPORT PREPARED")
    print("=" * 55)

    print("Your maintenance report has been prepared successfully.")

    print("\nIncident ID:", report["Incident_ID"])
    print("Detection status:", report["Detection_Status"])
    print("Network area:", report["Network_Area"])
    print("Alarm status:", report["Alarm_Status"])
    print("Recommended action:", report["Recommended_Action"])
    print("Report status:", report["Report_Status"])

    print("\nReport saved successfully:")
    print(report_path)

    print("\nThe report is ready for the maintenance team.")
    print("It still needs to be sent to the responsible personnel.")

    print("=" * 55)


# 9. CHATBOT CLASSIFICATION

def get_intent(user_input):

    user_vector = vectorizer.transform([user_input.lower()])

    if user_vector.nnz == 0:
        return None

    probabilities = chat_model.predict_proba(user_vector)[0]

    if max(probabilities) < 0.35:
        return None

    return chat_model.classes_[probabilities.argmax()]


# 10. CHATBOT LOGIC

def chat(user_input):

    load_detection_status()

    words = user_input.lower().replace("!", "").replace("?", "").split()

    if words and words[0] in ("hi", "hello", "hey"):
        return greeting_response

    if "thanks" in words or "thank" in words:
        return thanks_response

    if "status" in words and (
        "detection" in words or "network" in words
    ):
        show_detection_status()
        return "The latest detection information is displayed above."

    if "alert" in words or "report" in words or "notify" in words:
        create_maintenance_report()
        return "Maintenance report request completed."

    if "detected" in words and latest_detection["status"] == "LEAK":
        return (
            "CURRENT LEAK ALERT\n"
            "----------------------------------------\n"
            "The detection system is reporting a possible significant leak.\n"
            "I recommend notifying the responsible operator and "
            "arranging a maintenance investigation.\n\n"
            "You can ask me to prepare a maintenance alert."
        )

    intent = get_intent(user_input)

    if intent is None:
        return unrelated_response

    if intent == "normal_status":
        show_detection_status()
        return responses[intent]

    if intent == "maintenance_alert":
        create_maintenance_report()
        return "Maintenance report request completed."

    if intent == "leak_detected":
        if latest_detection["status"] == "LEAK":
            show_detection_status()

        return responses[intent]

    return responses[intent]



# 11. START THE CHATBOT

def start_assistant():

    print("\n" + "=" * 60)
    print("              WATER LEAK ASSISTANT CHATBOT")
    print("=" * 60)

    print("\nHello! Welcome to the Water Leak Assistant.")

    print(
        "\nI can help you with:"
        "\n- Water leak detection"
        "\n- Maintenance alerts"
        "\n- Pressure and flow problems"
        "\n- Leak prevention"
        "\n- Water conservation"
        "\n- Emergency guidance"
        "\n- How the detection system works"
    )

    print(
        "\nExamples:"
        "\n  has a leak been detected"
        "\n  what should i do if a leak is detected"
        "\n  prepare a maintenance report"
        "\n  why is water pressure dropping"
        "\n  what does unusual water flow mean"
        "\n  how can we prevent water leaks"
        "\n  how can we save water"
        "\n  how does the detection system work"
        "\n  show me the latest detection status"
    )

    print("\nType 'exit' at any time to close the chatbot.")
    print("=" * 60)

    while True:

        user_question = input("\nYou: ").strip()

        if user_question.lower() in ("exit", "quit", "bye"):
            print("\nThank you for using the Water Leak Assistant!")
            print("Stay alert and help prevent water losses.")
            break

        if user_question == "":
            print("Please enter a question.")
            continue

        response = chat(user_question)

        print("\nWater Leak Assistant:")
        print(response)

        print("\n" + "=" * 60)
        print("You can type another question or type 'exit' to close.")
        print("=" * 60)


# 12. PROGRAM ENTRY POINT

if __name__ == "__main__":
    start_assistant()
