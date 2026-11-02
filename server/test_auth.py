import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class AuthFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        database = Path(cls.temp.name) / "auth.db"
        os.environ["VAS_DATABASE_URL"] = f"sqlite:///{database.as_posix()}"
        os.environ["VAS_CREATE_SCHEMA"] = "true"
        os.environ["VAS_DEV_AUTO_VERIFY"] = "true"
        import server.app
        cls.module = importlib.reload(server.app)
        cls.client_context = TestClient(cls.module.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        cls.module.engine.dispose()
        cls.temp.cleanup()

    def test_register_login_refresh_replay_and_logout(self) -> None:
        email = "owner@example.com"
        password = "a long local test password"
        registered = self.client.post("/v1/auth/register", json={
            "email": email,
            "display_name": "Owner",
            "password": password,
        })
        self.assertEqual(registered.status_code, 202)

        logged_in = self.client.post("/v1/auth/login", json={
            "email": email,
            "password": password,
            "device_name": "Windows test device",
        })
        self.assertEqual(logged_in.status_code, 200)
        first = logged_in.json()

        profile = self.client.get("/v1/me", headers={"Authorization": f"Bearer {first['access_token']}"})
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["display_name"], "Owner")

        rotated = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(rotated.status_code, 200)
        second = rotated.json()
        self.assertNotEqual(second["refresh_token"], first["refresh_token"])

        replay = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(replay.status_code, 401)
        family_revoked = self.client.get("/v1/me", headers={"Authorization": f"Bearer {second['access_token']}"})
        self.assertEqual(family_revoked.status_code, 401)

        logged_in_again = self.client.post("/v1/auth/login", json={
            "email": email,
            "password": password,
            "device_name": "Windows test device",
        }).json()
        logged_out = self.client.post(
            "/v1/auth/logout",
            headers={"Authorization": f"Bearer {logged_in_again['access_token']}"},
        )
        self.assertEqual(logged_out.status_code, 204)
        after_logout = self.client.get(
            "/v1/me",
            headers={"Authorization": f"Bearer {logged_in_again['access_token']}"},
        )
        self.assertEqual(after_logout.status_code, 401)

    def test_login_error_does_not_reveal_account_existence(self) -> None:
        unknown = self.client.post("/v1/auth/login", json={
            "email": "missing@example.com",
            "password": "wrong",
            "device_name": "test",
        })
        known = self.client.post("/v1/auth/login", json={
            "email": "owner@example.com",
            "password": "wrong",
            "device_name": "test",
        })
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(known.status_code, 401)
        self.assertEqual(unknown.json()["code"], known.json()["code"])


if __name__ == "__main__":
    unittest.main()
