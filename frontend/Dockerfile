FROM node:20-alpine AS deps
WORKDIR /srv
COPY frontend/package.json ./package.json
COPY frontend/package-lock.json ./package-lock.json
RUN npm ci --no-audit --no-fund

FROM node:20-alpine AS builder
WORKDIR /srv
ARG NEXT_PUBLIC_BACKEND_URL
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL
COPY --from=deps /srv/node_modules ./node_modules
COPY frontend/. ./
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /srv
ENV NODE_ENV=production
COPY --from=builder /srv/.next ./.next
COPY --from=builder /srv/node_modules ./node_modules
COPY --from=builder /srv/package.json ./package.json
COPY --from=builder /srv/public ./public
COPY --from=builder /srv/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "start"]
