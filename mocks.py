class MockFile:
    def to_dict(self):
        return {'type': 'file', 'url': 'mock_url'}

class MockFileManager:
    def get_attr(self, file, attr):
        return f"mock_{attr}"

class MockVariableFactory:
    def segment_to_variable(self, segment, selector):
        return segment # Simplified for now

    def build_segment(self, value):
        return value # Simplified for now

class MockEncrypter:
    def obfuscated_token(self, token):
        return "********"

file_manager = MockFileManager()
variable_factory = MockVariableFactory()
encrypter = MockEncrypter()
