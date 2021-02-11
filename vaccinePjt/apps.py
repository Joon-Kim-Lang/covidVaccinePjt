from django.apps import AppConfig


class VaccinepjtConfig(AppConfig):
    name = 'vaccinePjt'

    def ready(self):
        #여기서 초기화 모델 불러오면 됨
        pass
