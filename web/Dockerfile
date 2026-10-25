FROM node:22-alpine AS build
WORKDIR /w
COPY web/package.json web/package-lock.json ./
COPY web/client/package.json client/package.json
COPY web/widget/package.json widget/package.json
COPY web/dashboard/package.json dashboard/package.json
RUN npm ci
COPY web/ ./
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /w/dashboard/.next/standalone ./
COPY --from=build /w/dashboard/.next/static ./dashboard/.next/static
COPY --from=build /w/dashboard/public ./dashboard/public
USER node
EXPOSE 3000
CMD ["node", "dashboard/server.js"]
