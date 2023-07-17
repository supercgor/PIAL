
def getLogger(path):
    import logging
    import logging.handlers
    from importlib import reload
    reload(logging)
    # Handler
    file_handler = logging.handlers.TimedRotatingFileHandler(path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Formatter
    color = {"GREEN" : '\x1b[32;20m', "BLUE" : '\x1b[34;20m', "GRAY" : '\x1b[49;90m', "RESET" : '\x1b[0m'}
    console_format = 'BLUE{name}RESET - GRAY{asctime}RESET - GREEN{levelname}RESET: {message}'
    for c in color:
        console_format = console_format.replace(c, color[c]) 
    file_handler.setFormatter(logging.Formatter(fmt='[{asctime} - {levelname}]: {message}', datefmt='%m/%d/%Y %H:%M:%S', style='{'))
    console_handler.setFormatter(logging.Formatter(fmt=console_format, datefmt='%H:%M:%S', style='{'))

    logging.basicConfig(level = logging.INFO,handlers=[file_handler, console_handler], encoding='utf-8')
    return logging.getLogger("Main")

if __name__ == "__main__":
    logger = get_logger("test.log")
    logger.info("Test")