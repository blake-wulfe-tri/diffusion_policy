from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NoOpEnvRunner(BaseImageRunner):
    def run(self, *args, **kwargs):
        return {}
