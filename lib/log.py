# log Module
import logging
import sys
from logging import Logger
from typing import Optional, Self, TypeAlias


class Formatter_(logging.Formatter):
    mod_width = 20

    def format(self, record):
        """override logging.Formatter.format"""
        func_lineno = record.funcName + ':' + str(record.lineno)
        func_lineno = func_lineno[-self.mod_width :].ljust(self.mod_width)
        return '%s %s %s %7s %s' % (
            record.name,
            self.formatTime(record, '%H:%M:%S'),
            func_lineno,
            record.levelname,
            record.msg,
        )


class Log(Logger):
    """Wrapper Class of logging.Logger

    Args:
        Logger (_type_): logging.Logger

    Returns:
        _type_: Log
    """

    FilePath: TypeAlias = str
    LoggerName: TypeAlias = str
    HandlerName: TypeAlias = str
    Handlers: TypeAlias = list[logging.Handler]
    Files: TypeAlias = list[tuple[HandlerName, FilePath]]

    # __format: str = '%(name)s %(asctime)s %(funcName)s %(levelname)7s %(message)s'
    __datefmt: str = '%H:%M:%S'
    __loggerName: set[LoggerName] = set()

    @staticmethod
    def __getHandlers(
        logger: Logger, handlerName: Optional[HandlerName] = None
    ) -> Handlers:
        """get a handler in logger

        Args:
            logger (Logger): logger object returned by Log.getLogger
            handlerName (Optional[HandlerName], optional): name of handler
            whose handler set to logger by setHandler. Defaults to None.

        Returns:
            Handlers: list of handlers indicated by handlerName.
            if handlerName is None (by default), this method returns all of handler
            set to logger.
        """
        if handlerName is None:
            hh = [h for h in logger.handlers]
        else:
            hh = [h for h in logger.handlers if h.name == handlerName]
        return hh

    @classmethod
    def __removeHandlers(
        cls, logger: Logger, handlerName: Optional[HandlerName] = None
    ) -> None:
        """remove handlers from logger

        Args:
            logger (Logger): logger returned from Log.getLogger
            handlerName (Optional[HandlerName], optional): name of handler
            whose handler set to logger by setHandler. Defaults to None.
            if handlerName is None, all of handlers set to logger are removed.
        """
        hh: Log.Handlers = cls.__getHandlers(logger, handlerName)
        for h in hh:
            logger.removeHandler(h)
            h.close()

    @classmethod
    def __setHandler(
        cls,
        logger: Logger,
        handlerName: HandlerName,
        filePath: Optional[FilePath] = None,
    ) -> None:
        """set a handler in logger.
        if handler name is 'null', remove all handlers from logger except null handler.
        if handler name is 'stderr', set stderr handler to logger

        Args:
            logger (Logger): logger returned from Log.getLogger
            handlerName (HandlerName): name of handler, which is removed from
            logger.
            filePath (Optional[FilePath], optional): path of log file. Defaults to None.
            if handler name is not neither 'null' nor 'stderr', set logfile handler,
            whose logfile name is set 'a.log' by default.
            if handlerName is 'null' or 'stderr', filepath is ignored.
        """
        h: logging.Handler
        if handlerName == 'null':
            cls.__removeHandlers(logger)
            h = logging.NullHandler()
        else:
            cls.__removeHandlers(logger, 'null')
            if len(cls.__getHandlers(logger, handlerName)) != 0:
                cls.__removeHandlers(logger, handlerName)

            if handlerName in 'stderr':
                # h = cls.__getHandlers(logger, handlerName)
                h = logging.StreamHandler(sys.stderr)
            else:
                if filePath is None:
                    filePath = 'a.log'
                h = logging.FileHandler(filePath, mode='a')

        h.set_name(handlerName)
        # h.setFormatter(logging.Formatter(cls.__format, cls.__datefmt))
        h.setFormatter(Formatter_())
        logger.addHandler(h)

    @classmethod
    def getLogfilePath(
        cls, logger: Logger, handlerName: Optional[HandlerName] = None
    ) -> Files:
        '''get filepath from logger with handler name'''
        ff: Log.Files
        if handlerName is None:
            ff = [
                (cls.HandlerName(h.name), h.__dict__['baseFilename'])
                for h in logger.handlers
                if 'baseFilename' in h.__dict__.keys()
            ]
        else:
            ff = [
                (cls.HandlerName(h.name), h.__dict__['baseFilename'])
                for h in logger.handlers
                if (h.name == handlerName and 'baseFilename') in h.__dict__.keys()
            ]
        return ff

    @classmethod
    def __reopenLogfile(
        cls, logger: Logger, handlerName: Optional[HandlerName] = None
    ) -> Files:
        """reopen log file

        Args:
            logger (Logger): logger returned from Log.getLogger
            handlerName (Optional[HandlerName], optional): name of handler
            whose handler set to logger by setHandler. Defaults to None.
            if handlerName is None, reopen all handlers set to logger.

        Returns:
            Files: list of log files reopened
        """
        ff: Log.Files = cls.getLogfilePath(logger, handlerName)
        for f in ff:
            cls.__removeHandlers(logger, f[0])
            cls.__setHandler(logger, *f)
        return ff

    @classmethod
    def setHandler(
        cls,
        logger: Logger,
        handlerName: HandlerName = 'stderr',
        filePath: Optional[FilePath] = None,
    ) -> None:
        """set handler to logger

        Args:
            logger (Logger): logger returned from Log.getLogger
            handlerName (Optional[HandlerName], optional): name of handler
            whose handler set to logger by setHandler. Defaults to None.
            if handlerName is None, reopen all handlers set to logger.
            filePath (Optional[FilePath], optional): path of log file.
            Defaults to None.
        """
        ff: Log.Files = cls.__reopenLogfile(logger, handlerName)
        if handlerName == 'stderr' and len(ff) != 0:
            return
        cls.__setHandler(logger, handlerName, filePath)

    @classmethod
    def resetHandler(cls, logger: Logger) -> None:
        """remove all handlers from logger and set logger to null

        Args:
            logger (Logger): logger returned from Log.getLogger
        """
        cls.__removeHandlers(logger)
        cls.__setHandler(logger, 'null')

    @classmethod
    def getLogger(cls, loggerName: LoggerName = __name__) -> Logger:
        """get logger from logging class and initialize it

        Args:
            loggerName (LoggerName, optional): name of logger. Defaults to __name__.

        Returns:
            Logger: logger instance created by logging
        """
        cls.__loggerName.add(loggerName)
        logger: Logger = logging.getLogger(loggerName)
        cls.__removeHandlers(logger)
        cls.__setHandler(logger, 'null')
        logger.setLevel(logging.NOTSET)
        logger.propagate = False
        return logger

    @classmethod
    def shutdown(cls, logger: Optional[Logger] = None) -> None:
        """remove all handlers from logger

        Args:
            logger (Optional[Logger], optional): logger returned from Log.getLogger.
            Defaults to None. if logger is None, remove all handlers from logger
        """
        if logger is None:
            for loggerName in sorted(cls.__loggerName):
                logger = logging.getLogger(loggerName)
                cls.__removeHandlers(logger)
            cls.__loggerName.clear()
        else:
            cls.__removeHandlers(logger)
            cls.__loggerName.remove(logger.name)

    @classmethod
    def status(cls) -> None:
        """inspect variables of this class for debugging"""
        print('<<< Log class variables >>>')
        for loggerName in sorted(cls.__loggerName):
            print(f'{loggerName}:')
            logger = logging.getLogger(loggerName)
            print('  handlers:')
            for h in logger.handlers:
                print(f'    {h.name} -> {h}')
            print('  logfiles:')
            ff = cls.getLogfilePath(logger)
            for f in ff:
                print(f'    {f[0]} -> {f[1]}')

    def __new__(cls) -> Self:  # type: ignore
        return super().__new__(cls)


if __name__ == '__main__':
    print(
        'running the following shell commands on your terminal is useful '
        'to watch logfile (a.log) at real time'
    )
    print('touch a.log; tail -f a.log')

    logger_name: Log.LoggerName = __name__
    logger: logging.Logger = Log.getLogger(logger_name)
    Log.setHandler(logger)
    Log.setHandler(logger, 'logfile', 'a.log')
    logger.setLevel(logging.DEBUG)
    logger.debug('debug message')
    logger.info('information')
    logger.error('error message')
    # logger.fatal('fatal message')
    # logger.critical('critical message')
    Log.status()
