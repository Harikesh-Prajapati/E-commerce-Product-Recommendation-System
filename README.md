# E-commerce-Product-Recommendation-System
E-commerce Product Recommendation System
A production-ready recommendation engine implementing core algorithms used by Amazon, Netflix, and major e-commerce platforms. Features multiple recommendation approaches with a real-time API.

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/ML-Recommendation%2520Systems-orange
https://img.shields.io/badge/License-MIT-lightgrey

✨ Features
Multiple Algorithm Support:

✅ User-based Collaborative Filtering

✅ Item-based Collaborative Filtering

✅ Hybrid Recommendation Approach

✅ Popular Items Fallback

Real-time API: HTTP server with GET/POST endpoints

Production Ready: Clean architecture, error handling, and validation

Realistic Data Generation: Synthetic e-commerce data with patterns

No Dependencies: Pure Python implementation

🚀 Quick Start
Run the Server
bash
python ecommerce_recommender.py
API Usage
bash
# Get recommendations for user 5
curl "http://localhost:8000/recommend?user_id=5&top_n=10"

# POST with JSON body
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 5, "top_n": 10, "method": "hybrid"}'
📊 API Endpoints
Method	Endpoint	Description
GET	/	API information
GET	/recommend	Get recommendations for user
POST	/recommend	Get recommendations with JSON body
GET	/stats	Get system statistics
🏗️ Project Structure
The system implements:

User-based Collaborative Filtering: Find similar users and recommend items they liked

Item-based Collaborative Filtering: Find similar items to those the user already liked

Hybrid Approach: Combine both methods for better recommendations

Popular Items Fallback: Recommended popular items when user data is insufficient

🔧 Technologies Used
Pure Python 3 (no external dependencies)

HTTP Server built on standard library

Cosine Similarity for user/item comparisons

Realistic Data Generation with user behavior patterns

📈 Business Impact
This system demonstrates technologies that:

Drive 35% of Amazon's revenue through recommendations

Increase conversion rates by 5-30% in e-commerce

Reduce bounce rates and improve user engagement

👨‍💻 Author
Harikesh Prajapati

Email: harikeshprajapati1242006@gmail.com

LinkedIn: Harikesh Prajapati

GitHub: [Your GitHub Profile]

📝 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Inspired by recommendation systems at Amazon, Netflix, and Spotify

Built with Python standard library for maximum compatibility

📞 Contact
If you have any questions, feel free to reach out!

Email: harikeshprajapati1242006@gmail.com

LinkedIn: Harikesh Prajapati

⭐ If you find this project useful, please give it a star on GitHub!

How to Run This Project
Save the code as ecommerce_recommender.py

Run with: python ecommerce_recommender.py

Open your browser to: http://localhost:8000

Test endpoints with curl or Postman

The server will start on port 8000 and provide recommendations for user IDs between 1-100 (from the generated sample data).

