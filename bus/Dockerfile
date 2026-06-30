FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /saha-bus ./main.go

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /saha-bus /saha-bus
EXPOSE 8090
ENTRYPOINT ["/saha-bus"]
