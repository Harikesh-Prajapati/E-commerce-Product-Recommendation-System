# E-commerce-Product-Recommendation-System
 🛒 E-commerce Product Recommendation System

A production-ready recommendation engine inspired by the systems behind Amazon, Netflix, and major e-commerce platforms.
This project implements multiple recommendation algorithms, provides a real-time API, and is built entirely with pure Python (no external dependencies).




---

✨ Features

Multiple Algorithms

✅ User-based Collaborative Filtering

✅ Item-based Collaborative Filtering

✅ Hybrid Recommendation Approach

✅ Popular Items Fallback

Real-time API

Lightweight HTTP server with GET/POST endpoints

Ready for integration into apps or dashboards

Production-Grade Design

Clean architecture, input validation, error handling

Scalable and easy to extend

Data Simulation

Synthetic e-commerce user behavior with realistic patterns

No Extra Dependencies

Runs on pure Python standard library


---


🚀 Quick Start
1. Run the Server
python ecommerce_recommender.py

2. API Usage
# Get recommendations for user 5
curl "http://localhost:8000/recommend?user_id=5&top_n=10"

# POST request with JSON body
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 5, "top_n": 10, "method": "hybrid"}'


---


📊 API Endpoints
Method	Endpoint	Description
GET	/	API information
GET	/recommend	Get recommendations for a user
POST	/recommend	Get recommendations (with JSON body)
GET	/stats	View system statistics


---


🏗️ Project Architecture

User-based CF → Finds similar users and suggests items they liked

Item-based CF → Recommends similar items to those a user already enjoyed

Hybrid → Blends both methods for stronger accuracy

Popular Fallback → Returns trending items when user history is limited

🔧 Tech Stack

Python 3.8+ (Standard Library only)

Cosine Similarity for user/item similarity

HTTP Server (via Python http.server)

Synthetic Data Generator for e-commerce behavior simulation

---


📈 Business Value

Recommendation engines power:

🛍️ 35%+ of Amazon’s revenue

📈 5–30% higher conversion rates in e-commerce

🎯 Better personalization, reduced bounce rate, higher engagement


---


👨‍💻 Author

Harikesh Prajapati

📧 ## Contact

💼 **LinkedIn**: [Your LinkedIn](https://www.linkedin.com/in/harikesh-prajapati-05034027a

💻 - **GitHub**: [Your GitHub](https://github.com/Harikesh-Prajapati)  

📝 License

This project is licensed under the MIT License
.

🙏 Acknowledgments

Inspired by Amazon, Netflix, and Spotify recommendation systems

Built with Python standard library for maximum portability

⭐ If you find this project useful, don’t forget to star the repo!
