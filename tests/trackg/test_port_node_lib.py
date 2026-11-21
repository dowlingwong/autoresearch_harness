"""The node-library copy rule of the port checks (port_support.node_lib_copy_ok): a node's ar_isolated.py is the
current master, or a declared earlier master version frozen in its pin; autoresearch_fixedtokens must carry the
current master (R1-TRACKG-AR-COND7 option A)."""
import subprocess

from .port_support import NODE_LIB_EARLIER, NODE_LIB_MASTER, ROOT, node_lib_copy_ok


def _earlier() -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", "c025f89a:src/autoresearch/trackg/node_lib/ar_isolated.py"],
                          capture_output=True, check=True).stdout


def test_node_lib_copy_rule():
    master = NODE_LIB_MASTER.read_bytes()
    assert node_lib_copy_ok("mlab_cifar10", master) and node_lib_copy_ok("autoresearch_fixedtokens", master)
    earlier = _earlier()
    import hashlib
    assert hashlib.sha256(earlier).hexdigest() in NODE_LIB_EARLIER
    assert node_lib_copy_ok("mlab_cifar10", earlier)
    assert not node_lib_copy_ok("autoresearch_fixedtokens", earlier)          # the AR node needs the current master
    assert not node_lib_copy_ok("mlab_cifar10", earlier + b"# edited\n")     # an undeclared version is refused
