type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';

class Logger {
  private static formatMessage(level: LogLevel, message: string, context?: Record<string, unknown>) {
    const timestamp = new Date().toISOString();
    const contextString = context ? ` | ${JSON.stringify(context)}` : '';
    return `[${timestamp}] [${level}] ${message}${contextString}`;
  }

  static info(message: string, context?: Record<string, unknown>) {
    console.log(this.formatMessage('INFO', message, context));
  }

  static warn(message: string, context?: Record<string, unknown>) {
    console.warn(this.formatMessage('WARN', message, context));
  }

  static error(message: string, context?: Record<string, unknown>) {
    console.error(this.formatMessage('ERROR', message, context));
  }

  static debug(message: string, context?: Record<string, unknown>) {
    if (process.env.NODE_ENV === 'development') {
      console.debug(this.formatMessage('DEBUG', message, context));
    }
  }
}

export default Logger;
