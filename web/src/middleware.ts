import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import Logger from './lib/logger';

export function middleware(request: NextRequest) {
  const requestId = Math.random().toString(36).substring(2, 10);
  const startTime = Date.now();
  const { pathname } = request.nextUrl;
  
  Logger.info(`Incoming request: ${request.method} ${pathname}`, { requestId });

  // Clone request headers and add our custom ID
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-request-id', requestId);

  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });

  // Performance Watchdog (log long-running requests)
  const duration = Date.now() - startTime;
  if (duration > 5000) {
    Logger.warn(`Slow request detected: ${pathname}`, { requestId, duration: `${duration}ms` });
  }

  // Add security headers
  const securityHeaders = {
    'X-DNS-Prefetch-Control': 'on',
    'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
    'X-XSS-Protection': '1; mode=block',
    'X-Frame-Options': 'SAMEORIGIN',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'origin-when-cross-origin',
    'x-request-id': requestId,
    'x-response-time': `${duration}ms`
  };

  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  return response;
}

export const config = {
  matcher: '/api/:path*',
};
