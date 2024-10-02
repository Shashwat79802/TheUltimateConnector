from crewai import Agent, Task, Crew, Process
from gemini import GEMINI_MODEL


def create_agent(user_profile):
    return Agent(
        role=f"AI Avatar for {user_profile['name']}",  # Simplify the role for easier reference
        goal="Converse with the AI Avatar of the other user and help in determining if establishing a connection with the other user would be helpful in terms of future growth, business, professional reasons, etc.",
        backstory=f"""I am an AI Avatar of {user_profile['name']} created to help them in determining if establishing a connection with the other user would be beneficial.
                      {user_profile['name']} is a {user_profile['profession']} based in {user_profile['location']} with {user_profile['experience']['years']} years of experience in the industry and is currently working as a {user_profile['experience']['current_role']}.
                      They have previously worked as a {', '.join(user_profile['experience']['previous_roles'])}.
                      They are skilled in {', '.join(user_profile['skills'])} and have worked on projects like {', '.join(user_profile['projects'])}.
                      They are interested in {', '.join(user_profile['interests'])} and their goals include {user_profile['goals']}.""",
        llm=GEMINI_MODEL,
        verbose=True,
        memory=False,
        additional_context={
            "user_profile": user_profile,
        },
        allow_delegation=False
    )

def initiate_conversation(user_a_profile, user_b_profile):
    # Create stateless agents with dynamic context injection
    agent_a = create_agent(user_a_profile)
    agent_b = create_agent(user_b_profile)

    # Define the task for conversation
    task1 = Task(
        # description="Communicate with the AI Avatar of the other user to determine if a connection is beneficial based on shared interests and skills. Then arrive at a mutual conclusion. And remember, everytime making a connection isn't necessary until there's a lot of scope for future collaboration between the two users",
        description="""The task is to converse with the AI Agent of other users and understand whether the users profile aligns with your interests and skills. Then, decide if you should connect with the other user based on shared interests and skills.
                       First, start by sharing about yourself and then ask questions to the AI Avatar of the other user to understand their profile. Based on the conversation, decide if you should connect with the other user or not.
                       The conversation should be exactly as the real users are talking to each other themselves. Remember to ask the right questions and provide the right information to make an informed decision.""",
        # expected_output="A decision on whether the two users should connect based on shared interests and skills. Release a json in this format {'decision': 'yes/no', 'reason': 'reason for the decision'}",
        expected_output="Upon the mutual discussion, tell the AI Avatar of the other user if you would like to connect with them or not. If yes, provide a reason for the same. If no, provide a reason for the same.",
        agent=agent_a,
    )

    task2 = Task(
        # description="Communicate with the AI Avatar of the other user to determine if a connection is beneficial based on shared interests and skills. Then arrive at a mutual conclusion. And remember, everytime making a connection isn't necessary until there's a lot of scope for future collaboration between the two users",
        # description="Communicate with the AI Avatar of the other user that's been trying to connect with you and answer their question, also, ask questions from your side too and determine if a connection is beneficial based on shared interests and skills. Then arrive at a mutual conclusion. And remember, everytime making a connection isn't necessary until there's a lot of scope for future collaboration between the two users",
        description=""""The task is to first understand about the other users profile that they're sharing and then if the user seems to be interesting and aligns well with your interests and skills, then you can go on with sharing about yourself and ask the right questions
                       to understand if the other user is a good fit for you to connect with. Based on the conversation, decide if you should connect with the other user or not. The conversation should be exactly as the real users are talking to each other themselves.""",
        # expected_output="A decision on whether the two users should connect based on shared interests and skills. Release a json in this format {'decision': 'yes/no', 'reason': 'reason for the decision'}",
        expected_output="Upon the mutual discussion, understand if the AI Avatar of the other user is worth connecting take and share your views and upon final discussion and decision, share a json in this format {'decision': 'yes/no', 'reason': 'reason for the decision'}",
        agent=agent_b,
    )

    # Create a crew to facilitate the conversation
    crew = Crew(
        agents=[agent_a, agent_b],
        tasks=[task1, task2],
        process=Process.sequential,  # Run tasks in sequence
        verbose=False
    )

    # Start the conversation
    return crew.kickoff()


if __name__ == "__main__":
    user_profile_1 = {
        "name": "Alice Johnson",
        "age": 28,
        "location": "San Francisco, CA",
        "profession": "Software Engineer",
        "skills": ["Python", "JavaScript", "React", "Node.js"],
        "interests": ["AI", "Open Source", "Hackathons", "Tech Blogging"],
        "github": "https://github.com/alicejohnson",
        "linkedin": "https://www.linkedin.com/in/alicejohnson",
        "projects": ["AI-Powered Chatbot", "Open Source Contributor - React"],
        "experience": {
            "years": 5,
            "current_role": "Senior Engineer at TechCorp",
            "previous_roles": ["Frontend Developer at StartupX", "Backend Engineer at DevHub"],
        },
        "goals": "Looking to collaborate on open-source AI projects."
    }

    user_profile_2 = {
        "name": "Brian Thompson",
        "age": 35,
        "location": "New York, NY",
        "profession": "Digital Marketing Manager",
        "skills": ["SEO", "Content Strategy", "Google Analytics", "Social Media"],
        "interests": ["Brand Building", "Content Marketing", "Data-Driven Marketing"],
        "github": None,
        "linkedin": "https://www.linkedin.com/in/brianthompson",
        "projects": ["Growth Hacking Campaign", "Lead Generation Optimization"],
        "experience": {
            "years": 8,
            "current_role": "Marketing Lead at AdGen",
            "previous_roles": ["SEO Specialist at MarketMax", "Content Manager at AdGrow"],
        },
        "goals": "Seeking partnerships in content marketing and brand strategy."
    }

    user_profile_3 = {
        "name": "Michael Rodriguez",
        "age": 40,
        "location": "Chicago, IL",
        "profession": "Mechanical Engineer",
        "skills": ["AutoCAD", "SolidWorks", "Thermodynamics", "Structural Analysis"],
        "interests": ["Automotive Design", "Energy Efficiency", "3D Printing"],
        "github": None,
        "linkedin": "https://www.linkedin.com/in/michaelrodriguez",
        "projects": ["Automotive Suspension Redesign", "Wind Turbine Efficiency Improvements"],
        "experience": {
            "years": 15,
            "current_role": "Lead Mechanical Engineer at AutoTech",
            "previous_roles": ["Senior Engineer at BuildCorp", "Project Engineer at EngiWorks"],
        },
        "goals": "Looking to collaborate on mechanical engineering projects and automotive design."
    }

    conversation_history_1 = []
    conversation_history_2 = []

    # Initiate the conversation between the two users
    result = initiate_conversation(user_profile_1, user_profile_3)
    print(result)
