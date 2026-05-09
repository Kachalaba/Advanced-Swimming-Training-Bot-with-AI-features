FROM node:20-alpine AS deps
WORKDIR /srv
COPY frontend/package.json ./package.json
RUN npm install --no-audit --no-fund

FROM node:20-alpine AS builder
WORKDIR /srv
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
