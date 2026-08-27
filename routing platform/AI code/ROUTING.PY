department_map = {
    "Flooding": "Drainage Department",
    "Drainage": "Drainage Department",
    "Road Damage": "Roads Department",
    "Garbage": "Sanitation Department",
    "Streetlight": "Electrical Department",
    "Water Supply": "Water Department",
    "Electricity": "Electricity Department",
    "Traffic": "Traffic Department",
    "Infrastructure Damage": "Municipal Engineering",
    "Public Safety": "Safety Department"
}
#severity rules
priority_map = {
    "Critical": "Emergency",
    "High": "High",
    "Medium": "Normal",
    "Low": "Low"
}
#coinfident therehold
CONFIDENCE_THRESHOLD = 0.80
#routing function
def route_complaint(category, severity, confidence):

    department = department_map.get(category)

    priority = priority_map.get(severity)

    if department is None:
        return {
            "department": None,
            "priority": priority,
            "routing_status": "HUMAN_REVIEW",
            "needs_human_review": True
        }

    if confidence >= CONFIDENCE_THRESHOLD:
        routing_status = "AUTO_ROUTE"
        needs_human_review = False
    else:
        routing_status = "HUMAN_REVIEW"
        needs_human_review = True

    return {
        "department": department,
        "priority": priority,
        "routing_status": routing_status,
        "needs_human_review": needs_human_review
    }
result = route_complaint(
    "Flooding",
    "High",
    0.81
)

print(result)

