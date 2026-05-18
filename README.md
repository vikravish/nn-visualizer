Real-time neural network training visualizer built on a from-scratch NumPy MLP, trained on the 70,000-image MNIST database (via scikit).
I made this as a learning tool to make understand the math behind neural networks, propagation, gradient descent, etc easier to actually visualize.

# Backend
cd backend && pip install fastapi uvicorn scikit-learn numpy
uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm run dev -- -p 3001

Live-streaming training metrics via SSE
Weight distributions, gradient norms, activation sparsity updating each epoch
Post-training confusion matrix and gallery of misclassified digits

The MLP was built fully from scratch in NumPy, no Tensor or PyTorch or other frames/libs. Backprop, ReLU, Softmax, and cross-entropy loss all hard-coded with detailed comments about the importance of each.

Will (soon) incorporate more advanced neural networks via PyTorch

Python · NumPy · FastAPI · Next.js · TypeScript · Recharts · D3 · Tailwind CSS
