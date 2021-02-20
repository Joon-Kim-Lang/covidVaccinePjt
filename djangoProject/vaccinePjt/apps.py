from django.apps import AppConfig
import vaccinePjt.autocrawling as ac
import vaccinePjt.pfizermodeling as pfm
import vaccinePjt.modernamodeling as mfm
import threading
import time


class VaccinepjtConfig(AppConfig):
    name = 'vaccinePjt'

    def ready(self):
        start = time.time()
        pfm.forPfizerInit()
        # mfm.forModernaInit()
        print(time.time() - start)
        _thread = threading.Thread(target=ac.auto_crawling)
        _thread.setDaemon(True)
        _thread.start()

        pass
