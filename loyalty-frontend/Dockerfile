FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build Next.js app (optional for dev, required for prod)
RUN npm run build

# Expose port
EXPOSE 3000

# Start command
CMD ["npm", "run", "dev"]
