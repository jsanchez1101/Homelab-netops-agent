# Lab Topology — "OSPF Triangle" on VyOS / EVE-NG

3 VyOS routers, full mesh, OSPF area 0. SW1 is gone — it added nothing the
agent could meaningfully query, and VyOS isn't a switch anyway.

## Diagram

```
                Cloud0 (pnet0 → node2 NIC → your LAN)
              ┌──────────┬──────────┬──────────┐
            eth0       eth0       eth0            (mgmt, passive in OSPF)
         ┌────┴───┐  ┌───┴────┐  ┌───┴────┐
         │   R1   │  │   R2   │  │   R3   │
         │1.1.1.1 │  │2.2.2.2 │  │3.3.3.3 │
         └─┬────┬─┘  └─┬────┬─┘  └─┬────┬─┘
           │    │      │    │      │    │
          eth1 eth2   eth1 eth2   eth1 eth2
           │    │      │    │      │    │
           └────┼──────┘    └──────┘    │
        10.0.12.0/30      10.0.23.0/30  │
        R1=.1  R2=.2      R2=.1  R3=.2  │
                │                       │
                └───────────────────────┘
                      10.0.13.0/30
                    R1=.1      R3=.2
```

## Addressing

| Link    | Subnet        | A              | B              |
|---------|---------------|----------------|----------------|
| R1–R2   | 10.0.12.0/30  | R1 eth1 = .1   | R2 eth1 = .2   |
| R2–R3   | 10.0.23.0/30  | R2 eth2 = .1   | R3 eth1 = .2   |
| R1–R3   | 10.0.13.0/30  | R1 eth2 = .1   | R3 eth2 = .2   |
| lo      | /32           | 1.1.1.1 / 2.2.2.2 / 3.3.3.3 |
| eth0    | YOUR LAN      | R1=.151  R2=.152  R3=.153   |

Configs use **192.168.86.x** for mgmt (based on node1 sitting at
192.168.86.55). VERIFY first — on node1: `ip route | grep default`. If your
LAN is a different subnet, find/replace in configs/ and devices.yaml. Make
sure .151–.153 are outside the DHCP pool.

## Ground truth baked into evals/tasks.yaml

- R1 OSPF neighbors: **2**
- DR on R1–R2 segment: **R2** (equal priority, higher router-id)
- OSPF routes in R2's table: **3** (1.1.1.1/32, 3.3.3.3/32, 10.0.13.0/30)
- R1 → 2.2.2.2: yes, next hop **10.0.12.2**
- Hello/dead timers R1↔R2: defaults both sides → **10 / 40**

## One-time: install the VyOS image into EVE-NG

1. Download a VyOS rolling ISO from https://vyos.net/get/nightly-builds/
   (rolling is the free one; any recent 1.5/current build is fine).
2. Copy to node2 and stage it (SSH to EVE-NG as root):
   ```bash
   scp vyos-*.iso root@100.67.189.100:/tmp/
   ssh root@100.67.189.100
   mkdir -p /opt/unetlab/addons/qemu/vyos-current
   cd /opt/unetlab/addons/qemu/vyos-current
   mv /tmp/vyos-*.iso cdrom.iso
   qemu-img create -f qcow2 virtioa.qcow2 2G
   /opt/unetlab/wrappers/unl_wrapper -a fixpermissions
   ```
3. In the EVE-NG web UI: add a node → template **VyOS** (now selectable),
   set "number of nodes" = 3, names R1/R2/R3. RAM 1024 MB each is plenty.
4. Boot ONE node, console in (login **vyos / vyos**), run:
   ```
   install image
   ```
   Accept defaults, set a password when asked (use vyos), poweroff. This
   writes VyOS onto the node's virtual disk so config persists.
   Repeat for each node (each has its own disk). After all three are
   installed you can optionally delete cdrom.iso from the template dir so
   future boots skip the live CD.

## Build the lab

1. Add a **Network** object → type **Management (Cloud0)**.
2. Cable (nodes must be stopped): each router's **eth0 → Cloud0**;
   R1 eth1 ↔ R2 eth1; R2 eth2 ↔ R3 eth1; R1 eth2 ↔ R3 eth2.
3. Start all nodes, console into each, log in (vyos / vyos — or the password
   you set during install image), and paste its file from `configs/`.
   The files are full `set` command blocks ending in `commit` + `save`.

## Verify from node1

```bash
ping 192.168.86.151
ssh admin@192.168.86.151      # password <PASSWORD>
show ip ospf neighbor          # R1: two neighbors in Full state
show ip route ospf             # on R2: three routes
```
Give OSPF ~40s after the last router boots. If SSH works manually, the
agent will work.

## Checklist before evals

- [ ] ssh admin@ each of .151/.152/.153 from node1
- [ ] R1 shows 2 Full neighbors
- [ ] R2 shows 3 OSPF routes
- [ ] devices.yaml IPs match reality
