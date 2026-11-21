"""The node-library copy rule of the port checks (port_support.node_lib_copy_ok): a node's ar_isolated.py is the
current master, or a declared earlier master version frozen in its pin and allowed for that node by name;
autoresearch_fixedtokens must carry the current master (R1-TRACKG-AR-COND7 option A)."""
import hashlib

from .port_support import NODE_LIB_EARLIER, NODE_LIB_MASTER, ROOT, node_lib_copy_ok


def test_node_lib_copy_rule():
    master = NODE_LIB_MASTER.read_bytes()
    assert node_lib_copy_ok("mlab_cifar10", master) and node_lib_copy_ok("autoresearch_fixedtokens", master)
    earlier = (ROOT / "nodes" / "mlab_cifar10" / "ar_isolated.py").read_bytes()     # the copy frozen in its pin
    assert earlier != master and hashlib.sha256(earlier).hexdigest() in NODE_LIB_EARLIER
    assert node_lib_copy_ok("mlab_cifar10", earlier) and node_lib_copy_ok("mlgym_houseprice", earlier)
    assert not node_lib_copy_ok("autoresearch_fixedtokens", earlier)          # the AR node needs the current master
    assert not node_lib_copy_ok("future_node", earlier)                       # only the seven named nodes
    assert not node_lib_copy_ok("mlab_cifar10", earlier + b"# edited\n")     # an undeclared version is refused
